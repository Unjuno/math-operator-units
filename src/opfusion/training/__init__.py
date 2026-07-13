from .config import OptimizerConfig, RecoveryConfig, RunConfig, load_run_config
from .data import (
    EXPERIMENT_OPERATORS,
    EncodedTrainingExample,
    SyntheticDataConfig,
    SyntheticTraceFactory,
    TrainingExample,
)
from .design_config import ModelDesignConfig, load_design_run_config, load_model_design, model_design
from .strict_verifier import install_strict_verifier
from .design_controls import install_model_design_controls

# Install one strict verifier for typed and surface profiles. Keeping the
# verifier shared prevents evaluation semantics from changing across the main
# surface condition and the typed diagnostic ablation.
install_strict_verifier()
# Model-design controls wrap the already normalized/shared-prefix generator.
install_model_design_controls()

__all__ = [
    "OptimizerConfig",
    "RecoveryConfig",
    "RunConfig",
    "load_run_config",
    "ModelDesignConfig",
    "load_design_run_config",
    "load_model_design",
    "model_design",
    "EXPERIMENT_OPERATORS",
    "SyntheticDataConfig",
    "SyntheticTraceFactory",
    "TrainingExample",
    "EncodedTrainingExample",
]
