from dataclasses import dataclass

from model.norm import NormType


@dataclass
class EncoderConf:
    n_filters: int
    encoder_kernel_size: int
    encoder_stride: int


@dataclass
class MaskNetConf:
    bottleneck_dim: int
    hidden_dim: int
    skip_dim: int
    kernel_size: int
    nb_layers: int
    nb_conv_blocks_per_layer: int
    norm_type: NormType


@dataclass
class TrainingConf:
    epochs: int
    batch_size: int
    num_workers: int
    half_lr: bool
    early_stop: bool
    seed: int | None


@dataclass
class QuantizeConf:
    epochs: int


@dataclass
class OptimConf:
    # We use Adam
    # optimizer: str
    lr: float
    weight_decay: float


@dataclass
class DataConf:
    task: str
    sample_rate: int
    n_src: int
    segment: int
    metadata_dir: str


class Conf:
    encoder: EncoderConf
    masknet: MaskNetConf
    training: TrainingConf
    quantize: QuantizeConf
    optim: OptimConf
    data: DataConf
    exp_dir: str

    def __init__(
        self,
        data: dict,
    ):
        self.encoder = EncoderConf(**data["encoder"])
        self.masknet = MaskNetConf(
            bottleneck_dim=data["masknet"]["bottleneck_dim"],
            hidden_dim=data["masknet"]["hidden_dim"],
            skip_dim=data["masknet"]["skip_dim"],
            kernel_size=data["masknet"]["kernel_size"],
            nb_layers=data["masknet"]["nb_layers"],
            nb_conv_blocks_per_layer=data["masknet"]["nb_conv_blocks_per_layer"],
            norm_type=NormType(data["masknet"]["norm_type"]),
        )
        self.training = TrainingConf(**data["training"])
        if "quantize" in data:
            self.quantize = QuantizeConf(**data["quantize"])
        self.optim = OptimConf(**data["optim"])
        self.data = DataConf(**data["data"])
        self.exp_dir: str = data["exp_dir"]
