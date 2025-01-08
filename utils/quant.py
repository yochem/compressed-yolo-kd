from torch import nn

def total_qbits(model):
    def recursive_walk(module):
        qbits = []

        if isinstance(module, nn.Module):
            if hasattr(module, "qbits") and callable(getattr(module, "qbits")):
                qbits.append(module.qbits())

            for child in module.children():
                qbits.extend(recursive_walk(child))

        return qbits

    return recursive_walk(model)
