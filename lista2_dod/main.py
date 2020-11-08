if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista2_dod'

from matplotlib import pyplot as plt

from lista2_dod.reg_mlp import DropPerceptron, RegularizedPerceptron

if __name__ == "__main__":
    # model = DropPerceptron(0.7)
    model = RegularizedPerceptron('l12', lambda1=.025, lambda2=.025)
    logger = model.train(True)

    logs = logger.get_logs()
    acc = logs['accuracies']
    test_error = logs['test_errors']
    train_error = logs['train_errors']
    print(f'Best accuracy: {round(max(acc) * 100, 2)}%')
    plt.plot(test_error, label='test')
    plt.plot(train_error, label='train')
    plt.xlabel('Test')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
