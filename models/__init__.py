# models/__init__.py
__all__ = ["Evo1Model", "CaduceusModel", "DNABERTModel", "DNABERT2Model", "DNABERTSModel", "HyenaDNAModel", "HyenaDNALocal", "NucleotideTransformerModel", "Evo2Model", "NemotronHModel", "NemotronHModelTT", "NemotronHMSAModel", "GPNMSAOriginalModel", "HyenaDNAFineTuner", "Heads", "MLPHead", "GPNMSAModel", "GenosModel", "GROVERModel", "GPNBrassicalesModel", "OmniRegGPTModel", "RNAFMModel", "RiNALMoModel", "ESMModel", "ESM2Model", "PhyschemDistillModel", "NemotronPoETStructv2Model"]

def __getattr__(name):
    if name == "Evo1Model":
        from .evo1_model import Evo1Model
        return Evo1Model
    if name == "CaduceusModel":
        from .caduceus_model import CaduceusModel
        return CaduceusModel
    if name == "DNABERTModel":
        from .dnabert_model import DNABERTModel
        return DNABERTModel
    if name == "DNABERT2Model":
        from .dnabert2_model import DNABERT2Model
        return DNABERT2Model
    if name == "DNABERTSModel":
        from .dnaberts_model import DNABERTSModel
        return DNABERTSModel
    if name == "HyenaDNAModel":
        from .hyenadna_model import HyenaDNAModel
        return HyenaDNAModel
    if name == "HyenaDNALocal":
        from .hyenadna_local import HyenaDNALocal
        return HyenaDNALocal
    if name == "NucleotideTransformerModel":
        from .nucleotide_transformer_model import NucleotideTransformerModel
        return NucleotideTransformerModel

    if name == "Evo2Model":
        from .evo2_model import Evo2Model
        return Evo2Model

    if name == "NemotronHModel":
        from .nemotronH_model import NemotronHModel
        return NemotronHModel
    if name == "NemotronHModelTT":
        from .nemotronH_model_padding import NemotronHModelTT
        return NemotronHModelTT

    if name == "NemotronHMSAModel":
        from .nemotronH_MSA_model import NemotronHMSAModel
        return NemotronHMSAModel
    
    if name == "HyenaDNAFineTuner":
        from .hyenadna_finetune import HyenaDNAFineTuner
        return HyenaDNAFineTuner
    if name == "Heads":
        from .heads import Heads
        return Heads
    if name == "MLPHead":
        from .mlp_head import MLPHead
        return MLPHead
    if name == "GPNMSAModel":
        from .gpn_msa_model import GPNMSAModel
        return GPNMSAModel
    if name == "GenosModel":
        from .genos_model import GenosModel
        return GenosModel
    if name == "GROVERModel":
        from .grover_model import GROVERModel
        return GROVERModel
    if name == "GPNBrassicalesModel":
        from .gpn_brassicales_model import GPNBrassicalesModel
        return GPNBrassicalesModel
    if name == "OmniRegGPTModel":
        from .omnireg_model import OmniRegGPTModel
        return OmniRegGPTModel
    if name == "RNAFMModel":
        from .rnafm_model import RNAFMModel
        return RNAFMModel
    if name == "RiNALMoModel":
        from .rinalmo_model import RiNALMoModel
        return RiNALMoModel
    if name == "ESMModel":
        from .esm_model import ESMModel
        return ESMModel
    if name == "ESM2Model":
        from .esm2_model import ESM2Model
        return ESM2Model
    if name == "PhyschemDistillModel":
        from .physchem_distill import PhyschemDistillModel
        return PhyschemDistillModel
    if name == "NemotronPoETStructv2Model":
        from .nemotron_poetstructv2_model import NemotronPoETStructv2Model
        return NemotronPoETStructv2Model
    raise AttributeError(name)
