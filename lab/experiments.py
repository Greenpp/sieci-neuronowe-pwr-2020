import os
import pickle
from datetime import datetime
from typing import Any, List, Tuple, Union

import numpy as np


class Experiment:
    def __init__(
        self,
        model: type,
        title: str,
        repetitions: int,
        test_parameter: Tuple[str, Union[List, Tuple[Any, Any, Any]]],
        results_dir: str = 'wyniki',
        f_name: str = None,
        **kwargs,
    ) -> None:
        self.title = title
        self.model = model
        self.repetitions = repetitions

        f_name = f'{datetime.now():%Y%m%d%H%M%S}' if f_name is None else f_name
        self.result_path = f'{os.getcwd()}/{results_dir}/{f_name}.pkl'

        self.results = {'title': title, 'results': dict()}

        self.parameters = kwargs
        self.test_param_name, test_param_values = test_parameter
        self.results['var'] = self.test_param_name
        if type(test_param_values) is tuple:
            # Create list of tested values
            test_param_values = self._unpack_range(test_param_values)

        # Max test value length for printing
        self.max_test_val_len = max(map(lambda x: len(str(x)), test_param_values))
        self.test_param_values = test_param_values

    def run(self) -> None:
        for test_val in self.test_param_values:
            test_param = {self.test_param_name: test_val}
            self.results['results'][test_val] = []
            for i in range(self.repetitions):
                print(
                    f'Tested value: {test_val:{self.max_test_val_len}} | {i+1}/{self.repetitions}',
                    end='\r',
                )
                model = self.model(**self.parameters, **test_param)
                logger = model.train()

                self.results['results'][test_val].append(logger.get_logs())
            # Done need to override whole count
            done_fill_size = len(f'{self.repetitions}/{self.repetitions}')
            print(
                f'Tested value: {test_val:{self.max_test_val_len}} | {"Done":<{done_fill_size}}'
            )

        self._save_result()

    def _unpack_range(self, param_range: Tuple) -> List:
        return [round(v, 8) for v in np.arange(*param_range)]

    def _save_result(self):
        with open(self.result_path, 'wb') as f:
            pickle.dump(self.results, f, protocol=pickle.HIGHEST_PROTOCOL)
