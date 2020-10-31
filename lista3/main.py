if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista3'


from matplotlib import pyplot as plt

from lista3.optim_mlp import OptimMNISTMLP

model = OptimMNISTMLP(
    activation_name='relu',
    optimizer_name='adam',
    initializer_name='he',
)
logger = model.train(verbose=True)

acc = logger.get_logs()['accuracies']
print(f'Best accuracy: {max(acc) * 100:.2f}%')
plt.plot(acc)
plt.show()
