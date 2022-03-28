import numpy as np
import torch
import matplotlib.pyplot as plt

def img01(x):
    x_min = np.amin(x)
    x_max = np.amax(x)
    return (x-x_min)/(x_max-x_min)

def get_hist(ds, num_classes):
    coco = ds.coco
    hist = np.zeros(num_classes)
    for i in range(num_classes):
        hist[i] = len(coco.getAnnIds(catIds=i))

    return hist


def plot_fig(x, fig_title):

    if torch.is_tensor(x):
        x = x.permute(1, 2, 0).numpy()

    img = img01(x)
    plt.imshow(img)
    plt.title(fig_title)

def plot_history(history):
    
    loss = history['train_loss']
    val_loss = history['val_loss']
    acc = history['train_acc']
    val_acc = history['val_acc']
    epochs_range = range(1, len(acc)+1)

    fig = plt.figure(figsize=(10,10))
    plt.subplot(211)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.subplot(212)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    plt.close(fig)
    plt.savefig("history.png")

def get_accuracy(true, pred):

    true = torch.argmax(true, dim=-1)
    pred = torch.argmax(pred, dim=-1)
    correct = (true == pred).type(torch.float).sum()
    batch_size = true.shape[0]
    return correct / batch_size
    