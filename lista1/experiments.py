if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista1'


from lab.experiments import Experiment
from lab.lab import Lab

from lista1.adaline import ANDAdaline
from lista1.perceptron import ANDPerceptron

if __name__ == "__main__":
    REPS = 100

    lab = Lab()

    ## PERCEPTRON
    # Theta
    perceptron_theta_uni = Experiment(
        title='Perceptron, wpływ theta, uni',
        repetitions=REPS,
        model=ANDPerceptron,
        test_parameter=('theta', [-1, -0.5, 0, 0.5, 1]),
        f_name='per_theta_uni',
        bipolar=False,
        bias=False,
        weight_range=(-0.2, 0.2),
        alpha=0.01,
    )
    lab.add_experiment(perceptron_theta_uni)
    perceptron_theta_bi = Experiment(
        title='Perceptron, wpływ theta, uni',
        repetitions=REPS,
        model=ANDPerceptron,
        test_parameter=('theta', [-1, -0.5, 0, 0.5, 1]),
        f_name='per_theta_bi',
        bipolar=True,
        bias=False,
        weight_range=(-0.2, 0.2),
        alpha=0.01,
    )
    lab.add_experiment(perceptron_theta_bi)

    # Weight range
    perceptron_weight_uni = Experiment(
        title='Perceptron, wpływ wag, uni',
        repetitions=REPS,
        model=ANDPerceptron,
        test_parameter=(
            'weight_range',
            [(0, 0), (-0.1, 0.1), (-0.2, 0.2), (-0.5, 0.5), (-0.8, 0.8), (-1, 1)],
        ),
        f_name='per_w_uni',
        bipolar=False,
        bias=True,
        theta=0,
        alpha=0.01,
    )
    lab.add_experiment(perceptron_weight_uni)
    perceptron_weight_bi = Experiment(
        title='Perceptron, wpływ wag, bi',
        repetitions=REPS,
        model=ANDPerceptron,
        test_parameter=(
            'weight_range',
            [(0, 0), (-0.1, 0.1), (-0.2, 0.2), (-0.5, 0.5), (-0.8, 0.8), (-1, 1)],
        ),
        f_name='per_w_bi',
        bipolar=True,
        bias=True,
        theta=0,
        alpha=0.01,
    )
    lab.add_experiment(perceptron_weight_bi)

    # Alpha
    perceptron_alpha_uni = Experiment(
        title='Perceptron, wpływ alpha, uni',
        repetitions=REPS,
        model=ANDPerceptron,
        test_parameter=('alpha', [0.0001, 0.001, 0.01, 0.1, 1]),
        f_name='per_alpha_uni',
        bipolar=False,
        bias=True,
        weight_range=(-0.5, 0.5),
        theta=0,
    )
    lab.add_experiment(perceptron_alpha_uni)
    perceptron_alpha_bi = Experiment(
        title='Perceptron, wpływ alpha, bi',
        repetitions=REPS,
        model=ANDPerceptron,
        test_parameter=('alpha', [0.0001, 0.001, 0.01, 0.1, 1]),
        f_name='per_alpha_bi',
        bipolar=True,
        bias=True,
        weight_range=(-0.5, 0.5),
        theta=0,
    )
    lab.add_experiment(perceptron_alpha_bi)

    ## ADALINE
    # Weight range
    adaline_weight = Experiment(
        title='Adaline, wpływ wag',
        repetitions=REPS,
        model=ANDAdaline,
        test_parameter=(
            'weight_range',
            [(0, 0), (-0.1, 0.1), (-0.2, 0.2), (-0.5, 0.5), (-0.8, 0.8), (-1, 1)],
        ),
        f_name='ada_w',
        bias=True,
        theta=0,
        alpha=0.01,
        epsilon=0.2,
    )
    lab.add_experiment(adaline_weight)

    # Alpha
    adaline_alpha = Experiment(
        title='Adaline, wpływ alpha',
        repetitions=REPS,
        model=ANDAdaline,
        test_parameter=('alpha', [0.0001, 0.001, 0.01, 0.1, 1]),
        f_name='ada_alpha',
        bias=True,
        weight_range=(-0.5, 0.5),
        theta=0,
        epsilon=0.2,
    )
    lab.add_experiment(adaline_alpha)

    # Epsilon
    adaline_epsilon = Experiment(
        title='Adaline, epsilon',
        repetitions=1,
        model=ANDAdaline,
        test_parameter=('epsilon', [0]),
        f_name='ada_epsilon',
        bias=True,
        weight_range=(-0.5, 0.5),
        theta=0,
        alpha=0.01,
        fail_after_max_epochs=False,
    )
    lab.add_experiment(adaline_epsilon)

    lab.run()
