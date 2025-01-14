from torch import nn


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


def size_per_layer(model: nn.Module) -> list[int]:
    # return [model_size(layer) for layer in next(model.children())]
    return [sum(total_qbits(layer).item()) for layer in next(model.children())]

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
