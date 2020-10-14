if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista1'


from lista1.adaline import ANDAdaline
from lista1.perceptron import ANDPerceptron


def print_label(label: str) -> None:
    print(f'{10*"="}{label.upper()}{10*"="}')


def adaline_test() -> None:
    print_label('adaline')
    model = ANDAdaline(
        theta=0, bias=True, weight_range=(-0.5, 0.5), alpha=0.01, epsilon=0.2
    )
    logger = model.train()
    print(f'Epochs: {logger.get_logs()["epochs"]}')


def perceptron_unipolar_test():
    print_label('perceptron unipolar')
    model = ANDPerceptron(
        bipolar=False, theta=0, bias=True, weight_range=(-0.5, 0.5), alpha=0.01
    )
    logger = model.train()
    print(f'Epochs: {logger.get_logs()["epochs"]}')


def perceptron_bipolar_test():
    print_label('perceptron bipolar')
    model = ANDPerceptron(
        bipolar=True, theta=0, bias=True, weight_range=(-0.5, 0.5), alpha=0.01
    )
    logger = model.train()
    print(f'Epochs: {logger.get_logs()["epochs"]}')


if __name__ == '__main__':
    perceptron_unipolar_test()
    perceptron_bipolar_test()
    adaline_test()
