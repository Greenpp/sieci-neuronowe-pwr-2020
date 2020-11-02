if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista2'


from lab.experiments import Experiment
from lab.lab import Lab

from lista2.mlp import MNISTMLP


def experiments() -> None:
    REPS = 10

    lab = Lab(
        MNISTMLP,
        REPS,
        'lista2/wyniki',
        hidden_size=128,
        batch_size=32,
        weights_range=(-0.5, 0.5),
        alpha=0.01,
        activation_name='relu',
    )

    # Hidden size
    h_size = Experiment(
        title='h_size',
        f_name='h_size_1024',
        test_parameter=(
            'hidden_size',
            [
                16,
                128,
                512,
                2048,
            ],
        ),
    )
    lab.add_experiment(h_size)
    # Alpha
    alpha = Experiment(
        title='alpha',
        f_name='alpha_1024',
        test_parameter=(
            'alpha',
            [
                0.0001,
                0.001,
                0.01,
                0.1,
                1.0,
            ],
        ),
    )
    lab.add_experiment(alpha)
    # Weight range
    w_range = Experiment(
        title='w_range',
        f_name='w_range_1024',
        test_parameter=(
            'weights_range',
            [
                (0.0, 0.0),
                (-0.1, 0.1),
                (-0.5, 0.5),
                (-2.0, 2.0),
            ],
        ),
    )
    lab.add_experiment(w_range)
    # Batch
    batch_size = Experiment(
        title='batch_size',
        f_name='batch_size_1024',
        test_parameter=(
            'batch_size',
            [
                1,
                8,
                32,
                128,
                1024,
            ],
        ),
    )
    lab.add_experiment(batch_size)
    # ReLU
    activation = Experiment(
        title='activation',
        f_name='activation_1024',
        test_parameter=(
            'activation_name',
            [
                'sigmoid',
                'relu',
            ],
        ),
    )
    lab.add_experiment(activation)

    lab.run()


if __name__ == "__main__":
    experiments()
