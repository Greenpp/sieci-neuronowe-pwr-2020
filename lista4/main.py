if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista4'


from matplotlib import pyplot as plt

from lista4.conv import ConvMNIST

if __name__ == "__main__":
    model = ConvMNIST(kernel_size=4)
    logger = model.train(verbose=True)

    acc = logger.get_logs()['accuracies']
    print(f'Best accuracy: {max(acc) * 100:.2f}%')
    plt.plot(acc)
    plt.show()
