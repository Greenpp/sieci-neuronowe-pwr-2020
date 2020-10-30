import os
import pickle as pkl

from lab.experiments import Experiment


class Lab:
    def __init__(
        self, model_class: type, reps: int, results_dir: str, **kwargs
    ) -> None:
        self.model_class = model_class
        self.reps = reps
        self.results_dir = results_dir
        self.default_params = kwargs
        self.experiments = []

        self.separator_size = 20

    def add_experiment(self, experiment: Experiment) -> None:
        self.experiments.append(experiment)

    def run(self) -> None:
        self._check_if_results_dir_exists()
        for exp in self.experiments:
            self._print_title(exp.title)
            log = exp.run(self.model_class, self.reps, **self.default_params)
            self._print_exp_end()

            file_path = f'./{self.results_dir}/{exp.f_name}.pkl'
            with open(file_path, 'wb') as f:
                pkl.dump(log, f, pkl.HIGHEST_PROTOCOL)
        self._print_done()

    def _print_title(self, title) -> None:
        print(self.separator_size * '=')
        print(f'Testing: {title}')
        print(self.separator_size * '-')

    def _print_exp_end(self) -> None:
        print(self.separator_size * '-')

    def _print_done(self) -> None:
        print(self.separator_size * '=')
        print(self.separator_size * '=')
        print('All done')

    def _check_if_results_dir_exists(self) -> None:
        if not os.path.isdir(f'./{self.results_dir}'):
            raise Exception('Results directory doesn\'t exist')
