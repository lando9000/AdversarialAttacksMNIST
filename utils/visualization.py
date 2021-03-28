import matplotlib.pyplot as plt
import numpy as np


def printAccuracy(eps, acc, names):
    # Visualization
    plt.figure(figsize=(5, 10))
    plt.plot(eps, acc[0], label=names[0])
    plt.plot(eps, acc[1], label=names[1])
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()


def printLoss(eps, loss, names):
    plt.figure(figsize=(5, 10))
    plt.plot(eps, loss[0], label=names[0])
    plt.plot(eps, loss[1], label=names[1])
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Loss vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()
