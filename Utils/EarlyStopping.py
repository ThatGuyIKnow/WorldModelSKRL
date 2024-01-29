class EarlyStopping:
    def __init__(self, objective : ['min', 'max'], patience : int = 1, min_delta : float = 0.):
        self.objective = objective
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self._stop = False

    def step(self, validation_loss) -> bool:
        if self.objective == 'max':
            validation_loss *= -1
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self._stop = True
        return self._stop

    @property
    def stop(self):
        return self._stop
