
import shutil
import torch


def save_checkpoint(accelerator, state, is_best, filename="checkpoint.pth.tar"):
    if accelerator is not None and not accelerator.is_main_process:
        return None
    torch.save(state, filename)
    if is_best:
        best_filename = filename.replace(filename.split('\\')[-1], "checkpoint_best_loss.pth.tar")
        shutil.copyfile(filename, best_filename)


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count