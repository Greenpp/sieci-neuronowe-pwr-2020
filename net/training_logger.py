class TrainingLogger:
    def __init__(self) -> None:
        self.train_errors = []
        self.test_errors = []
        self.accuracies = []
        self.failed = False

    def log_train_error(self, error: float) -> None:
        self.train_errors.append(error)

    def log_test_error(self, error: float) -> None:
        self.test_errors.append(error)

    def log_accuracy(self, acc: float) -> None:
        self.accuracies.append(acc)

    def log_fail(self) -> None:
        self.failed = True

    def get_logs(self) -> dict:
        return {
            'train_errors': self.train_errors,
            'test_errors': self.test_errors,
            'batches': len(self.accuracies)
            - 1,  # first acc log is before training loop
            'accuracies': self.accuracies,
            'failed': self.failed,
        }
