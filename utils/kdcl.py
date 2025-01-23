import torch
import torch.functional as F


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(models, optimizers, train_loader, T, alpha):
    acc_recorder_list = []
    loss_recorder_list = []
    for model in models:
        model.train()
        acc_recorder_list.append(AverageMeter())
        loss_recorder_list.append(AverageMeter())

    for i, (imgs, label) in enumerate(train_loader):
        # torch.Size([batch, num_model, 3, 32, 32]) torch.Size([batch])
        outputs = torch.zeros(
            size=(len(models), imgs.size(0), 100), dtype=torch.float
        ).cuda()
        out_list = []
        # forward
        for model_idx, model in enumerate(models):

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                label = label.cuda()

            out = model.forward(imgs[:, model_idx, ...])
            outputs[model_idx, ...] = out
            out_list.append(out)

        # backward
        stable_out = outputs.mean(dim=0)
        stable_out = stable_out.detach()

        for model_idx, model in enumerate(models):
            ce_loss = F.cross_entropy(out_list[model_idx], label)
            div_loss = (
                F.kl_div(
                    F.log_softmax(out_list[model_idx] / T, dim=1),
                    F.softmax(stable_out / T, dim=1),
                    reduction="batchmean",
                )
                * T
                * T
            )

            loss = (1 - alpha) * ce_loss + (alpha) * div_loss

            optimizers[model_idx].zero_grad()
            if model_idx < len(models) - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()

            optimizers[model_idx].step()

            loss_recorder_list[model_idx].update(loss.item(), n=imgs.size(0))
            acc = accuracy(out_list[model_idx], label)[0]
            acc_recorder_list[model_idx].update(acc.item(), n=imgs.size(0))

    losses = [recorder.avg for recorder in loss_recorder_list]
    acces = [recorder.avg for recorder in acc_recorder_list]
    return losses, acces


def backward(models, outputs, optims, targets, alpha, T):
    outputs = torch.stack(outputs)
    stable_out = outputs.mean(dim=0)
    stable_out = stable_out.detach()

    losses = []
    accurs = []
    retain_graphs = (True, False)

    for model, optim, output, retain_graph in zip(
        models, optims, outputs, retain_graphs
    ):
        ce_loss = F.cross_entropy(output, targets)
        lsm = F.log_softmax(output / T, dim=1)
        sm = F.softmax(stable_out / T, dim=1)
        div_loss = F.kl_div(lsm, sm, reduction="batchmean") * T * T

        loss = (1 - alpha) * ce_loss + (alpha) * div_loss

        optim.zero_grad()
        loss.backward(retain_graph=retain_graph)
        optim.step()

        loss_meter = AverageMeter()
        loss_meter.update(loss.item(), n=len(targets))
        losses.append(loss_meter)

        acc = accuracy(output, targets)[0]
        acc_meter = AverageMeter()
        acc_meter.update(acc.item(), n=len(targets))
        accurs.append(acc_meter)

    avg_losses = [recorder.avg for recorder in losses]
    avg_accurs = [recorder.avg for recorder in accurs]
    return avg_losses, avg_accurs
