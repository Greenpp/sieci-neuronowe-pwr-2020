if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista2_dod'


from lab.experiments import Experiment
from lab.lab import Lab

from lista2_dod.reg_mlp import DropPerceptron, RegularizedPerceptron

if __name__ == "__main__":
    REPS = 10

    reg_lab = Lab(
        RegularizedPerceptron,
        REPS,
        'lista2_dod/wyniki',
    )

    exp_l = Experiment(
        title='L',
        f_name='l',
        test_parameter=(
            'regularization',
            ['l1', 'l2', 'l12'],
        ),
    )
    reg_lab.add_experiment(exp_l)

    drop_lab = Lab(
        DropPerceptron,
        REPS,
        'lista2_dod/wyniki',
    )

    drop_exp = Experiment(
        title='Drop',
        f_name='drop',
        test_parameter=(
            'drop_rate',
            [0, 0.2, 0.5, 0.8],
        ),
    )
    drop_lab.add_experiment(drop_exp)

    reg_lab.run()
    drop_lab.run()