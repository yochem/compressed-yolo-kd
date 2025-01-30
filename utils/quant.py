import json

from torch import nn

from utils.general import LOGGER


def total_qbits(model: nn.Module):
    def recursive_walk(module):
        qbits = []

        if isinstance(module, nn.Module):
            if hasattr(module, "qbits") and callable(getattr(module, "qbits")):
                qbits.append(module.qbits())
            else:
                for child in module.children():
                    qbits.extend(recursive_walk(child))

        return qbits

    return recursive_walk(model)


def layer_qbits(model: nn.Module) -> list[int]:
    return [
        round(sum([x.item() for x in total_qbits(layer)]))
        for layer in next(model.children())
    ]


def layer_size(model: nn.Module) -> list[int]:
    return [model_size(layer) for layer in next(model.children())]


def model_size(model: nn.Module) -> int:
    """In bytes."""

    def recursive_walk(module: nn.Module) -> list[float]:
        bits = []

        if isinstance(module, nn.Module):
            if hasattr(module, "qbits") and callable(getattr(module, "qbits")):
                bits.append(module.qbits().detach().item())
            else:
                # torch default 32 bits
                bits.append(
                    32 * sum(p.numel() for p in module.parameters(recurse=False))
                )

                for child in module.children():
                    bits.extend(recursive_walk(child))

        return bits

    return sum(recursive_walk(model)) // 8


class JsonResults:
    def __init__(self, path: str, model_params: dict):
        self.path = path
        self.model_params = model_params
        self.epoch_fields = [
            "epoch",
            "precision",
            "recall",
            "size_total",
            "loss_object",
            "loss_class",
            "loss_bbox",
            "loss_imitation",
            "loss_compression",
        ]

        self.epoch_fields.extend(
            [f"size_l{i}" for i, _ in enumerate(self.model_params["layers"])]
        )
        self.epoch_fields.extend(
            [f"qbits_l{i}" for i, _ in enumerate(self.model_params["layers"])]
        )
        self.epochs: list[dict] = []

    @property
    def data(self) -> dict:
        return {
            "params": self.model_params,
            "epochs": self.epochs,
        }

    def write(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self.data, f)

    def add_epoch(self, epoch_data):
        # missing keys
        if len(diff := set(self.epoch_fields) - set(epoch_data)) > 0:
            LOGGER.warning(f'missing result fields: {", ".join(diff)}')
        # unknown keys
        if len(diff := set(epoch_data) - set(self.epoch_fields)) > 0:
            LOGGER.warning(f'unknown result fields: {", ".join(diff)}')

        self.epochs.append(epoch_data)
