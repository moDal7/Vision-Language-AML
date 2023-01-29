import matplotlib as plt

def plot_loss(train_loss: list, val_loss: list, iteration_log: list) -> None:
    plt.plot(iteration_log, train_loss, label="Training")
    plt.plot(iteration_log, val_loss, label="Validation")

    plt.title("Training - Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend() 