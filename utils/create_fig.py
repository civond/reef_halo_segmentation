import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

def create_fig(df, save_fig_path):
    x = range(len(df))

    ms = 3

    # Plot
    plt.figure(1, figsize=(5,4))

    #plt.axvline(x=len(x)-4, linestyle="--", color='k', label='Early Stop') 
    plt.plot(x, df['train_loss'], 'o-', markersize=ms, color='b', label='train loss')
    plt.plot(x, df['val_loss'], 's-', markersize=ms, color='r', label='val loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Mask R-CNN Train / Val Loss")
    plt.xlim(0,len(x)-1)
    plt.legend()


    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(save_fig_path)
    #plt.show()
