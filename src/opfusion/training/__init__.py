from .config import OptimizerConfig, RecoveryConfig, RunConfig, load_run_config
from .data import (
    EXPERIMENT_OPERATORS,
    EncodedTrainingExample,
    SyntheticDataConfig,
    SyntheticTraceFactory,
    TrainingExample,
)
from .strict_verifier import install_strict_verifier

# Install one strict verifier for typed and surface profiles. Keeping the
# verifier shared prevents evaluation semantics from changing across the main
# surface condition and the typed diagnostic ablation.
install_strict_verifier()

__all__ = [
    "OptimizerConfig",
    "RecoveryConfig",
    "RunConfig",
    "load_run_config",
    "EXPERIMENT_OPERATORS",
    "SyntheticDataConfig",
    "SyntheticTraceFactory",
    "TrainingExample",
    "EncodedTrainingExample",
]
