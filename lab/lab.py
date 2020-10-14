from lab.experiments import Experiment


class Lab:
    def __init__(self) -> None:
        self.experiments = []

    def add_experiment(self, experiment: Experiment) -> None:
        self.experiments.append(experiment)

    def run(self) -> None:
        for experiment in self.experiments:
            print(15 * '=')
            print(f'Testing: {experiment.title}')
            print(15 * '-')
            experiment.run()
        print(15 * '=')
        print(15 * '=')
        print('All done')
