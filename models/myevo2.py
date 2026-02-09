from functools import partial
import huggingface_hub
from huggingface_hub import snapshot_download, constants, hf_hub_download
import os
import pkgutil
import torch
from typing import List, Tuple, Dict, Union
import yaml


from vortex.model.generation import generate as vortex_generate
from vortex.model.model import StripedHyena
from vortex.model.tokenizer import CharLevelTokenizer
from vortex.model.utils import dotdict, print_rank_0, load_checkpoint

from evo2.scoring import score_sequences, score_sequences_rc

# 默认本地模型路径
DEFAULT_MODEL_BASE_PATH = '../../model_weight'

MODEL_NAMES = [
    'evo2_40b',
    'evo2_7b',
    'evo2_40b_base',
    'evo2_7b_base',
    'evo2_1b_base',
    'evo2_7b_262k',
    'evo2_7b_microviridae',
]

HF_MODEL_NAME_MAP = {
    'evo2_40b': 'arcinstitute/evo2_40b',
    'evo2_7b': 'arcinstitute/evo2_7b',
    'evo2_40b_base': 'arcinstitute/evo2_40b_base',
    'evo2_7b_base': 'arcinstitute/evo2_7b_base',
    'evo2_1b_base': 'arcinstitute/evo2_1b_base',
    'evo2_7b_262k': 'arcinstitute/evo2_7b_262k',
    'evo2_7b_microviridae': 'evo-design/evo-2-7b-8k-microviridae',
}

CONFIG_MAP = {
    'evo2_7b': 'configs/evo2-7b-1m.yml',
    'evo2_40b': 'configs/evo2-40b-1m.yml',
    'evo2_7b_base': 'configs/evo2-7b-8k.yml',
    'evo2_40b_base': 'configs/evo2-40b-8k.yml',
    'evo2_1b_base': 'configs/evo2-1b-8k.yml',
    'evo2_7b_262k': 'configs/evo2-7b-262k.yml',
    'evo2_7b_microviridae': 'configs/evo2-7b-8k.yml',
}


class Evo2:
    def __init__(self, model_name: str = MODEL_NAMES[1], local_path: str = None, use_local: bool = True):
        """
        Load an Evo 2 checkpoint.

        By default, uses local_path from DEFAULT_MODEL_BASE_PATH.
        If local_path is specified, uses that path directly.
        If use_local=False, falls back to HuggingFace download.

        Vortex automatically handles device placement on CUDA, and splits model across
        multiple GPUs if available.
        For models split across multiple GPUs, you can specify which GPUs to use with
        CUDA_VISIBLE_DEVICES. If using multi-gpu, do not use .to(device) manually.

        Notes:
        Evo 2 40b is too large to fit on a single H100 GPU, so needs multiple GPUs.
        You can change where HuggingFace downloads to by setting the HF_HOME environment
        variable.
        """
        if model_name not in MODEL_NAMES:
            raise ValueError(
                f'Invalid model name {model_name}. Should be one of: '
                f'{", ".join(MODEL_NAMES)}.'
            )

        # 获取当前文件所在目录，用于构建配置文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_relative_path = CONFIG_MAP[model_name]
        config_path_absolute = os.path.join(current_dir, config_relative_path)

        # 默认使用本地路径
        if use_local:
            if local_path is None:
                # 构建默认本地路径，尝试多种可能的路径
                model_filename = f"{model_name}.pt"
                # 尝试路径1: model_weight/model_name/model_name.pt
                local_path = os.path.join(DEFAULT_MODEL_BASE_PATH, model_name, model_filename)
                # 尝试路径2: model_weight/model_name.pt
                if not os.path.exists(local_path):
                    alt_path = os.path.join(DEFAULT_MODEL_BASE_PATH, model_filename)
                    if os.path.exists(alt_path):
                        local_path = alt_path
                # 尝试路径3: model_weight/model_name/*.pt (查找目录下的任何.pt文件)
                if not os.path.exists(local_path):
                    model_dir = os.path.join(DEFAULT_MODEL_BASE_PATH, model_name)
                    if os.path.isdir(model_dir):
                        for file in os.listdir(model_dir):
                            if file.endswith('.pt'):
                                local_path = os.path.join(model_dir, file)
                                break
            
            self.model = self.load_evo2_model(None, config_path_absolute, local_path)
        else:
            # 使用 HuggingFace 下载（需要相对路径用于 pkgutil）
            if local_path is not None:
                self.model = self.load_evo2_model(None, config_path_absolute, local_path)
            else:
                self.model = self.load_evo2_model(model_name, config_relative_path)
        
        self.tokenizer = CharLevelTokenizer(512)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        return_embeddings: bool = False,
        layer_names=None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with optional embedding extraction.
        
        Args:
            input_ids: Input token IDs
            return_embeddings: If True, returns embeddings from specified layers
            layer_names: List of layer names to extract embeddings from. Required if
                return_embeddings=True
            
        Returns:
            Tuple of (logits, embeddings_dict) if return_embeddings=True
            Tuple of (logits, None) otherwise
        """
        embeddings = {}
        handles = []
        
        if return_embeddings:
            if layer_names is None:
                raise ValueError(
                    "layer_names must be specified when return_embeddings=True. Look at "
                    "evo2_model.model.state_dict().keys() to see available layers."
                )
                
            def hook_fn(layer_name):
                def hook(_, __, output):
                    if isinstance(output, tuple):
                        output = output[0]
                    embeddings[layer_name] = output.detach()
                return hook
                
            # Register hooks for requested layers
            for name in layer_names:
                layer = self.model.get_submodule(name)
                handles.append(layer.register_forward_hook(hook_fn(name)))
        
        try:
            # Original forward pass
            with torch.no_grad():
                logits = self.model.forward(input_ids)
            
            if return_embeddings:
                return logits, embeddings
            return logits, None
            
        finally:
            for handle in handles:
                handle.remove()

    def __call__(self, input_ids, return_embeddings=False, layer_names=None):
        return self.forward(input_ids, return_embeddings, layer_names)

    def score_sequences(
        self,
        seqs: List[str],
        batch_size: int = 1,
        prepend_bos: bool = False,
        reduce_method: str = 'mean',
        average_reverse_complement: bool = False,
    ) -> List[float]:
        scoring_func = partial(
            score_sequences_rc if average_reverse_complement else score_sequences,
            model=self.model,
            tokenizer=self.tokenizer,
            batch_size=batch_size,
            prepend_bos=prepend_bos,
            reduce_method=reduce_method,
        )

        with torch.no_grad():
            try:
                scores = scoring_func(seqs)
            except Exception as e:
                raise RuntimeError(f"Error during sequence scoring: {str(e)}") from e

        return scores
    
    def generate(
        self,
        prompt_seqs: List[str],
        n_tokens: int = 500,
        temperature: float = 1.0,
        top_k: int = 4,
        top_p: float = 1.0,
        batched: bool = True,
        cached_generation: bool = True,
        verbose: int = 1,
        force_prompt_threshold: int = None,
    ) -> Tuple[List[str], List[float]]:
        """
        Generate sequences from a list of prompts.

        force_prompt_threshold: If specified, avoids OOM errors through teacher forcing if the prompt is longer than this threshold.

        If force_prompt_threshold is none, sets default assuming 1xH100 (evo2_7b) and 2xH100 (evo2_40b) to help avoid OOM errors.
        """

        with torch.no_grad():
            output = vortex_generate(
                prompt_seqs=prompt_seqs,
                model=self.model,
                tokenizer=self.tokenizer,
                n_tokens=n_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                batched=batched,
                cached_generation=cached_generation,
                verbose=verbose,
                force_prompt_threshold=force_prompt_threshold,
            )
            return output


    def load_evo2_model(
            self,
            model_name: str = MODEL_NAMES[1],
            config_path: str = None,
            local_path: str = None,
            remove_shards: bool = True,
    ):
        """
        Load HuggingFace checkpoint using StripedHyena 2.

        If local_path is specified, loads from local_path.
        Otherwise, downloads from HuggingFace.
        If remove_shards is True, removes HF checkpoint shards after merging to .pt file.
        """
        if local_path is not None:
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Model file not found at {local_path}")
            print(f"Loading model from {local_path}...")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
            print(f"Loading config from {config_path}...")
            with open(config_path, 'r') as f:
                config = dotdict(yaml.load(f, Loader=yaml.FullLoader))
            model = StripedHyena(config)
            load_checkpoint(model, local_path)
            return model
        
        hf_model_name = HF_MODEL_NAME_MAP[model_name]
        filename = f"{model_name}.pt"
        
        final_weights_path = os.path.join(os.path.dirname(constants.HF_HUB_CACHE), filename)
        if os.path.exists(final_weights_path):
            print(f"Found existing merged file: {final_weights_path}")
            weights_path = final_weights_path
            
            hf_hub_download(
                repo_id=hf_model_name, 
                filename="config.json"
            )
        else:
            repo_dir = snapshot_download(
                repo_id=hf_model_name,
            )
            
            # Check if the complete file already exists in the repo
            repo_weights_path = os.path.join(repo_dir, filename)
            if os.path.exists(repo_weights_path):
                print(f"Found complete file in repo: {filename}")
                weights_path = repo_weights_path
            else:
                print(f"Looking for checkpoint shards for {filename}")
                parts = []
                part_num = 0

                while True:
                    part_path = os.path.join(repo_dir, f"{filename}.part{part_num}")
                    if os.path.exists(part_path):
                        parts.append(part_path)
                        part_num += 1
                    else:
                        break
                
                if parts:
                    print(f"Found {len(parts)} shards, merging them...")
                    with open(final_weights_path, 'wb') as outfile:
                        for part in parts:
                            print(f"Merging shard: {os.path.basename(part)}")
                            with open(part, 'rb') as infile:
                                while True:
                                    chunk = infile.read(8192*1024)
                                    if not chunk: 
                                        break
                                    outfile.write(chunk)
                    
                    print(f"Successfully merged all shards into {final_weights_path}")
                    weights_path = final_weights_path
                    if remove_shards and os.path.exists(final_weights_path):
                        for part in parts:
                            real_path = os.path.realpath(part)
                            if os.path.exists(real_path):
                                os.remove(real_path)
                            if os.path.exists(part):
                                os.remove(part)
                else:
                    raise FileNotFoundError(f"Could not find {filename} or any of its shards in {repo_dir}")
                
        config = yaml.safe_load(pkgutil.get_data(__name__, config_path))
        global_config = dotdict(config, Loader=yaml.FullLoader)

        model = StripedHyena(global_config)
        load_checkpoint(model, weights_path)

        return model
