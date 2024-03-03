

from torch.optim import lr_scheduler

class ReduceLROnPlateau(lr_scheduler.ReduceLROnPlateau):
    def get_last_lr(self):
        return self._last_lr