if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista4'


from lab.experiments import Experiment
from lab.lab import Lab

from lista4.conv import ConvMNIST

if __name__ == "__main__":
    REPS = 10

    lab = Lab(
        ConvMNIST,
        REPS,
        'lista4/wyniki',
    )

    # Kernel size
    ker = Experiment(
        title='kernel',
        f_name='kernel',
        test_parameter=(
            'kernel_size',
            [
                3,
                5,
                7,
            ],
        ),
    )
    lab.add_experiment(ker)

    lab.run()