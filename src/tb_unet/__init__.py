"""TB U-Net: model, training, inference, and config utilities."""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "UNet":
        from tb_unet.model import UNet
        return UNet
    if name == "UNetWithAttention":
        from tb_unet.model import UNetWithAttention
        return UNetWithAttention
    if name == "build_unet":
        from tb_unet.model import build_unet
        return build_unet
    if name == "TBCellDataset":
        from tb_unet.dataset import TBCellDataset
        return TBCellDataset
    if name == "create_dataloaders":
        from tb_unet.dataset import create_dataloaders
        return create_dataloaders
    if name == "Trainer":
        from tb_unet.train import Trainer
        return Trainer
    if name == "TrainConfig":
        from tb_unet.config import TrainConfig
        return TrainConfig
    if name == "ModelConfig":
        from tb_unet.config import ModelConfig
        return ModelConfig
    if name == "predict":
        from tb_unet.inference import predict
        return predict
    if name == "load_model":
        from tb_unet.inference import load_model
        return load_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "UNet",
    "UNetWithAttention",
    "build_unet",
    "TBCellDataset",
    "create_dataloaders",
    "Trainer",
    "TrainConfig",
    "ModelConfig",
    "predict",
    "load_model",
]
