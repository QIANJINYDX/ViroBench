from abc import ABC, abstractmethod
from typing import List


class BaseModel(ABC):
    """
    模型基类 - 所有模型都要实现这个接口
    """

    def __init__(self, model_name: str, model_path: str):
        """
        Args:
            model_name: 模型名称，如 "evo2_7b"
            model_path: 模型文件路径
        """
        self.model_name = model_name
        self.model_path = model_path
        # 子类需要加载实际的模型和tokenizer

    # @abstractmethod
    # def score_sequences(self, sequences: List[str], batch_size: int = 256) -> List[float]:
    #     """
    #     对序列列表进行评分 - 核心接口

    #     Args:
    #         sequences: 待评分的序列列表，如 ["ATCG", "GCTA", ...]
    #         batch_size: 批处理大小

    #     Returns:
    #         每个序列的得分，如 [-2.34, -1.87, ...]

    #     注意：
    #         - 输出长度必须等于输入 sequences 的长度
    #         - 得分顺序必须与输入顺序一致
    #     """
    #     pass

    # @abstractmethod
    # def get_embedding(self, sequences: List[str], layer_name: str, batch_size: int = 64) -> List[float]:
    #     """
    #     获取序列的embedding
    #     """
    #     pass

    # @abstractmethod
    # def generate(
    #     self,
    #     prompt_seqs: List[str] = ["ACGT"],
    #     n_tokens: int = 400,
    #     temperature: float = 1.0,
    #     top_k: int = 4,
    # ) -> List[str]:
    #     """
    #     根据给定 prompt 序列生成新序列。

    #     Args:
    #         prompt_seqs: prompt 序列列表，如 ["ACGT", "TTAA", ...]
    #         n_tokens: 每条序列生成的 token 数
    #         temperature: 采样温度
    #         top_k: top-k 采样参数

    #     Returns:
    #         生成序列列表，长度必须等于输入 prompt_seqs 的长度，且顺序一致。
    #     """
    #     pass
