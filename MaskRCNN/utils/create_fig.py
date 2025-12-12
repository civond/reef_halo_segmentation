import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

def create_fig(df, save_fig_path, show_plot=False):
    x = range(len(df))

    ms = 3
    train_loss=df['train_loss']
    val_loss=df['val_loss']
    val_dice=df['val_dice']

    # Plot
    plt.figure(1, figsize=(5,7))
    plt.subplot(2,1,1)
    plt.plot(x, train_loss, 'o-', markersize=ms, color='b', label='train_loss')
    plt.plot(x, val_loss, 's-', markersize=ms, color='r', label='val_loss')
    plt.ylabel("Avg. Loss")
    plt.title("Loss")
    plt.xlim(0,len(x)-1)
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))


    plt.subplot(2,1,2)
    plt.plot(x, val_dice, 's-', markersize=ms, color='r', label='val_dice')
    plt.xlim(0,len(x)-1)
    plt.ylim(np.min(val_dice)-.1,1)
    plt.ylabel("Avg. Dice Score")
    plt.title("Dice Score")
    plt.xlabel("Epoch")
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(save_fig_path)

    if show_plot == True:
        plt.show()
