import matplotlib.pyplot as plt
import numpy as np


def plotAccuracy(eps, acc):
    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(eps, acc[0], label="LeNet")
    plt.plot(eps, acc[1], label="ResNet")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.show()


def plotLoss(eps, loss):
    plt.figure(figsize=(10, 5))
    plt.plot(eps, loss[0], label="LeNet")
    plt.plot(eps, loss[1], label="ResNet")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Loss vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()


def plotAdversarialExamples(eps, ex):
    cnt = 0
    plt.figure(figsize=(8, 10))
    for i in range(len(eps)):
        for j in range(len(ex[i])):
            cnt += 1
            plt.subplot(len(eps), len(ex[0]), cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel("Eps: {}".format(eps[i]), fontsize=14)
            orig, adv, img = ex[i][j]
            plt.title("{} -> {}".format(orig, adv))
            plt.imshow(img, cmap="gray")
    plt.tight_layout()
    plt.show()
