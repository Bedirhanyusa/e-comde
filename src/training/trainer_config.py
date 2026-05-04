from dataclasses import dataclass, field


@dataclass
class TrainerConfig:
    model_name: str = "dbmdz/bert-base-turkish-cased"
    num_labels: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    max_len: int = 128
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    seed: int = 42
    checkpoint_dir: str = "models/berturk_finetuned"
    label_names: list[str] = field(default_factory=lambda: ["olumsuz", "olumlu", "notr"])

    # Dusuk VRAM icin gradient accumulation (efektif batch = batch_size * accum_steps)
    gradient_accumulation_steps: int = 1
