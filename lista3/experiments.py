if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista3'


from lab.experiments import Experiment
from lab.lab import Lab

from lista3.optim_mlp import OptimMNISTMLP

if __name__ == "__main__":
    REPS = 10

    lab = Lab(
        OptimMNISTMLP,
        REPS,
        'lista3/wyniki',
        activation_name='relu',
        optimizer_name='sgd',
        initializer_name='range',
        alpha=0.01,
    )

    # Base
    base = Experiment(
        title='base',
        f_name='base',
        test_parameter=(
            'activation_name',
            [
                'relu',
                'sigmoid',
            ],
        ),
        alpha=0.1,
    )
    lab.add_experiment(base)

    # Optimizers
    opt_relu = Experiment(
        title='optimizers_relu',
        f_name='optimizers_relu',
        test_parameter=(
            'optimizer_name',
            [
                'sgd',
                'momentum',
                'nesterov',
                'adagrad',
                'adadelta',
                'adam',
            ],
        ),
    )
    lab.add_experiment(opt_relu)

    opt_sig = Experiment(
        title='optimizers_sig',
        f_name='optimizers_sig',
        test_parameter=(
            'optimizer_name',
            [
                'sgd',
                'momentum',
                'nesterov',
                'adagrad',
                'adadelta',
                'adam',
            ],
        ),
        activation_name='sigmoid',
    )
    lab.add_experiment(opt_sig)

    # Initializers
    init_relu = Experiment(
        title='init_relu',
        f_name='init_relu',
        test_parameter=(
            'initializer_name',
            [
                'xavier',
                'he',
            ],
        ),
        optimizer_name='adagrad',
    )
    lab.add_experiment(init_relu)

    init_sig = Experiment(
        title='init_sig',
        f_name='init_sig',
        test_parameter=(
            'initializer_name',
            [
                'xavier',
                'he',
            ],
        ),
        activation_name='sigmoid',
        optimizer_name='adam',
    )
    lab.add_experiment(init_sig)

    lab.run()
