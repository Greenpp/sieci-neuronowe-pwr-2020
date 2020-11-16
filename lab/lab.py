import os
from pathlib import Path
from typing import Any

from lab.experiments import Experiment
from lab.utils import merge_results, save_to_pkl


class Lab:
    def __init__(
        self, model_class: type, reps: int, results_dir: str, **kwargs: Any
    ) -> None:
        self.model_class = model_class
        self.reps = reps
        self.results_dir = f'./{results_dir}'
        self.tmp_dir = f'./{results_dir}/tmp'
        self.default_params = kwargs
        self.experiments = []

        self.separator_size = 20

    def add_experiment(self, experiment: Experiment) -> None:
        self.experiments.append(experiment)

    def run(self) -> None:
        self._check_if_results_dir_exists()
        self._create_tmp_dir()
        for exp in self.experiments:
            exp_name = exp.f_name
            tmp_files = []
            self._print_title(exp.title)
            for j, log in enumerate(
                exp.run(
                    self.model_class, self.reps, self.tmp_dir, **self.default_params
                )
            ):
                tmp_f_name = f'{exp_name}_s_{j}'
                save_to_pkl(tmp_f_name, self.tmp_dir, log)
                tmp_files.append(tmp_f_name)
            self._print_exp_end()

            exp_log = merge_results(tmp_files, self.tmp_dir)
            save_to_pkl(exp_name, self.results_dir, exp_log)
        self._print_done()

    def _print_title(self, title: str) -> None:
        print(self.separator_size * '=')
        print(f'Testing: {title}')
        print(self.separator_size * '-')

    def _print_exp_end(self) -> None:
        print(self.separator_size * '-')

    def _print_done(self) -> None:
        print(self.separator_size * '=')
        print(self.separator_size * '=')
        print('All done')

    def _create_tmp_dir(self) -> None:
        Path(self.tmp_dir).mkdir(exist_ok=True)

    def _check_if_results_dir_exists(self) -> None:
        if not os.path.isdir(self.results_dir):
            raise Exception('Results directory doesn\'t exist')
