from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainerState:
    """
    A class containing the [`Trainer`] inner state


    epoch (`int`, *optional*):
        Only set during training, will represent the epoch the training is at.

    global_step (`int`, *optional*, defaults to 0):
        During training, represents the number of update steps completed.

    log_history (`list[dict[str, float]]`, *optional*):
            The list of logs done since the beginning of training.

    best_metric (`float`, *optional*):
        When tracking the best model, the value of the best metric encountered so far.

    best_global_step (`int`, *optional*):
        When tracking the best model, the step at which the best metric was encountered.
        Used for setting `best_model_checkpoint`.
    """

    epoch: Optional[int] = None
    global_step: int = 0
    log_history: list[dict[str, float]] = None
    best_metric: Optional[float] = None
    best_global_step: Optional[int] = None

    def __post_init__(self):
        if self.log_history is None:
            self.log_history = []
