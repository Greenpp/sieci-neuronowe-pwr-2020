if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista1'


from lab.experiments import Experiment
from lab.lab import Lab

from lista1.adaline import ANDAdaline
from lista1.perceptron import ANDPerceptron


def experiments():
    REPS = 10

    ## PERCEPTRON
    perceptron_lab = Lab(
        ANDPerceptron,
        REPS,
        '/lista1/wyniki/test',
        bipolar=False,
        theta=0,
        bias=True,
        weight_range=(-0.5, 0.5),
        alpha=0.01,
    )

    # Theta
    perceptron_theta_uni = Experiment(
        title='Perceptron, uni, theta',
        f_name='per_theta_uni',
        test_parameter=('theta', [-1.0, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0]),
        bias=False,
    )
    perceptron_lab.add_experiment(perceptron_theta_uni)

    perceptron_theta_bi = Experiment(
        title='Perceptron, bi, theta',
        f_name='per_theta_bi',
        test_parameter=('theta', [-1.0, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0]),
        bias=False,
    )
    perceptron_lab.add_experiment(perceptron_theta_bi)

    # Weight range
    perceptron_w_uni = Experiment(
        title='Perceptron, uni, w',
        f_name='per_w_uni',
        test_parameter=(
            'weight_range',
            [
                (0.0, 0.0),
                (-0.1, 0.1),
                (-0.2, 0.2),
                (-0.5, 0.5),
                (-0.8, 0.8),
                (-1.0, 1.0),
            ],
        ),
    )
    perceptron_lab.add_experiment(perceptron_w_uni)

    perceptron_w_bi = Experiment(
        title='Perceptron, bi, theta',
        f_name='per_w_bi',
        test_parameter=(
            'weight_range',
            [
                (0.0, 0.0),
                (-0.1, 0.1),
                (-0.2, 0.2),
                (-0.5, 0.5),
                (-0.8, 0.8),
                (-1.0, 1.0),
            ],
        ),
    )
    perceptron_lab.add_experiment(perceptron_w_bi)

    # Alpha
    perceptron_alpha_uni = Experiment(
        title='Perceptron, uni, alpha',
        f_name='per_alpha_uni',
        test_parameter=('alpha', [0.0001, 0.001, 0.01, 0.1, 1.0]),
    )
    perceptron_lab.add_experiment(perceptron_alpha_uni)

    perceptron_alpha_bi = Experiment(
        title='Perceptron, bi, alpha',
        f_name='per_alpha_bi',
        test_parameter=('alpha', [0.0001, 0.001, 0.01, 0.1, 1.0]),
    )
    perceptron_lab.add_experiment(perceptron_alpha_bi)

    ## ADALINE
    adaline_lab = Lab(
        ANDAdaline,
        REPS,
        '/lista1/wyniki/test',
        theta=0,
        bias=True,
        weight_range=(-0.5, 0.5),
        alpha=0.01,
    )

    # Weight range
    adaline_w = Experiment(
        title='Adaline, w',
        f_name='ada_w',
        test_parameter=(
            'weight_range',
            [
                (0.0, 0.0),
                (-0.1, 0.1),
                (-0.2, 0.2),
                (-0.5, 0.5),
                (-0.8, 0.8),
                (-1.0, 1.0),
            ],
        ),
    )
    adaline_lab.add_experiment(adaline_w)

    # Alpha
    adaline_alpha = Experiment(
        title='Adaline, alpha',
        f_name='ada_alpha',
        test_parameter=('alpha', [0.0001, 0.001, 0.01, 0.1, 1.0]),
    )
    adaline_lab.add_experiment(adaline_alpha)

    # Epsilon
    adaline_epsilon = Experiment(
        title='Adaline, epsilon',
        f_name='ada_epsilon',
        test_parameter=('epsilon', [0]),
    )
    adaline_lab.add_experiment(adaline_epsilon)


    perceptron_lab.run()
    adaline_lab.run()


if __name__ == "__main__":
    experiments()