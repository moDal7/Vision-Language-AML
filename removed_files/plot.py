import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_loss(train_loss: list, val_loss: list, val_acc: list, iteration_log: list, experiment: str, target: str) -> None:
    
    train_loss = np.array(train_loss)
    val_loss =  np.array([elem.to("cpu") for elem in val_loss])
    val_acc = np.array(val_acc)
    iteration_log =  np.array(iteration_log)

    fig1, ax1 = plt.subplots()
    ax1.plot(iteration_log, train_loss, label="Training")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.grid()
    ax1.legend()
    fig1.savefig(f'./{experiment}_{target}_train_loss.png')

    fig2, ax2 = plt.subplots()
    ax2.plot(iteration_log, val_loss, label="Validation")
    ax2.set_title("Validation Loss")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.grid()
    ax2.legend()
    fig2.savefig(f'./{experiment}_{target}_validation_loss.png')

    fig3, ax3 = plt.subplots()
    ax3.plot(iteration_log, val_acc, label="Validation")
    ax3.set_title("Validation Accuracy")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Loss")
    ax3.grid()
    ax3.legend()
    fig3.savefig(f'./{experiment}_{target}_validation_accuracy.png')
    
