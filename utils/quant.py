from torch import nn


def total_qbits(model):
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


def size_per_layer(model: nn.Module):
    # return [sum(model_size(layer)) for layer in model.children()]
    # for l in model.children():
    #     print(l)
    #     print(model_size(l))
    #     print('--'*80)
    l = 0
    for _ in next(model.children()):
        l += 1
    print(l)
    exit(1)
    return s


def model_size(model):
    """In bytes."""

    def recursive_walk(module):
        bits = []

        if isinstance(module, nn.Module):
            if hasattr(module, "qbits") and callable(getattr(module, "qbits")):
                bits.append(module.qbits())
            else:
                # torch default 32 bits
                bits.append(
                    32 * sum(p.numel() for p in module.parameters(recurse=False))
                )

            for child in module.children():
                bits.extend(recursive_walk(child))

        return bits

    return sum(recursive_walk(model)) / 8
