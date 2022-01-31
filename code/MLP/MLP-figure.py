import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print("Wrong number of arguments! Usage: python", sys.argv[0], "<output-figure> <input-losses> [<other-input-losses> ...]")
    exit()

max_epochs = 0
for i in range(2, len(sys.argv)):
    log_data = pickle.load(open(sys.argv[i], "rb"))
    title = log_data["title"]
    train_losses = log_data["train_losses"]
    test_losses = log_data["test_losses"]
    assert(len(train_losses) == len(test_losses))
    n_epochs = len(train_losses)
    if n_epochs > max_epochs:
        max_epochs = n_epochs
    epochs = np.arange(n_epochs)
    plt.plot(epochs, train_losses, label=title + '-train')
    plt.plot(epochs, test_losses, label=title + '-test')

plt.legend()
plt.savefig(sys.argv[1])