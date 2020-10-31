class TrainingLogger:
    def __init__(self) -> None:
        self.train_errors = []
        self.test_errors = []
        self.accuracies = []
        self.batches = 0
        self.epochs = 0
        self.failed = False

    def log_train_error(self, error: float) -> None:
        self.train_errors.append(error)

    def log_test_error(self, error: float) -> None:
        self.test_errors.append(error)

    def log_accuracy(self, acc: float) -> None:
        self.accuracies.append(acc)

    def log_fail(self) -> None:
        self.failed = True

    def log_batch(self) -> None:
        self.batches += 1

    def log_epoch(self) -> None:
        self.epochs += 1

    def get_logs(self) -> dict:
        return {
            'train_errors': self.train_errors,
            'test_errors': self.test_errors,
            'batches': self.batches,
            'epochs': self.epochs,
            'accuracies': self.accuracies,
            'failed': self.failed,
        }
