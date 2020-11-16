from typing import Any, Generator, List, Tuple

from lab.utils import get_data, save_to_pkl


class Experiment:
    def __init__(
        self,
        title: str,
        f_name: str,
        test_parameter: Tuple[str, List],
        fail_after_limit: bool = False,
        **kwargs: Any,
    ) -> None:
        self.title = title
        self.f_name = f_name
        self.test_param_name, self.test_param_values = test_parameter
        self.fail_after_limit = fail_after_limit
        self.custom_params = kwargs

        self.skip_after_first_fails = 5
        self.first_failed = 0

    def run(
        self, model_class: type, reps: int, tmp_dir: str, **kwargs: Any
    ) -> Generator:
        val_keys = list(map(str, self.test_param_values))
        max_val_len = max(map(len, val_keys))
        reps_len = len(str(reps))

        for val, val_key in zip(self.test_param_values, val_keys):
            val_key = str(val)
            exp_log = dict()
            exp_log[val_key] = []

            params = kwargs.copy()
            test_param = {self.test_param_name: val}
            params.update(test_param)
            params.update(self.custom_params)

            tmp_files = []
            for i in range(reps):
                print(
                    f'\rTested value: {val_key:{max_val_len}}: {i+1:{reps_len}}/{reps}',
                    end='',
                )
                model = model_class(**params)
                training_logger = model.train(self.fail_after_limit)

                training_log = training_logger.get_logs()
                exp_log[val_key].append(training_log)

                if training_log['failed']:
                    if self.first_failed >= 0:
                        self.first_failed += 1
                    if self.first_failed >= self.skip_after_first_fails:
                        exp_log[val_key].clear()
                        break
                else:
                    self.first_failed = -1

                tmp_f_name = f'{val_key}_{i}'
                save_to_pkl(tmp_f_name, tmp_dir, training_log)
                tmp_files.append(tmp_f_name)

            data = get_data(tmp_files, tmp_dir)
            exp_log[val_key] = data

            if self.first_failed == -1:
                print(
                    f'\rTested value: {val_key:{max_val_len}}: Done{" " * reps_len * 2}'
                )
            else:
                print(
                    f'\rTested value: {val_key:{max_val_len}}: Skipped{" " * reps_len * 2}'
                )

            exp_log = {
                'title': self.title,
                'test_param': self.test_param_name,
                'results': exp_log,
            }

            yield exp_log
