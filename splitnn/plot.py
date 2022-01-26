import matplotlib.pyplot as plt


def plot_train(train_losses, train_accuracies, test_accuracies):
    """
    Args
    :param train_losses: historical train loss for each epoch
    :param train_accuracies: historical train accuracy for each epoch
    :param test_accuracies: historical test accuracy for each epoch
    :return:
    """
    plt.figure(figsize=(18, 3))
    plt.subplot(1, 3, 1)
    plt.plot(range(len(train_losses)), train_losses, '-ro')
    plt.title("Losses")
    plt.subplot(1, 3, 2)
    plt.plot(range(len(train_accuracies)), train_accuracies, '-ro')
    plt.title("Training Accuracy")
    plt.subplot(1, 3, 3)
    plt.plot(range(len(test_accuracies)), test_accuracies, '-ro')
    plt.title("Testing Accuracy")
    plt.savefig("../tmp/bank-splitnn.png")

