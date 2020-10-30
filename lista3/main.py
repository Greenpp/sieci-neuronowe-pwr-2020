if __name__ == '__main__' and __package__ is None:
    from os.path import dirname as dir
    from sys import path

    path.append(dir(path[0]))
    __package__ = 'lista3'


from matplotlib import pyplot as plt

from lista2.mnist_loader import MNISTLoader
from net.activations import ReLU, SoftmaxCE
from net.data_loader import DataLoader
from net.layers import FCLayer
from net.loss_functions import CrossEntropy
from net.model import Model
from net.trainers import AdamTrainer

loader = MNISTLoader()
tr, _, tes = loader.get_sets()

train_loader = DataLoader(tr)
test_loader = DataLoader(tes, None, False)

model = Model(FCLayer(784, 128, ReLU()), FCLayer(128, 10, SoftmaxCE()))

loss = CrossEntropy()
trainer = AdamTrainer(0.01, loss)
trainer.set_data_loaders(train_loader, test_loader)

trainer.train(model, max_batches=150, verbose=True)
logger = trainer.get_logger()

acc = logger.get_logs()['accuracies']
print(f'Best accuracy: {max(acc) * 100:.2f}%')
plt.plot(acc)
plt.show()
