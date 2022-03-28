import os
import argparse
import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import vit_b_16
import torch.optim as optim

from utils import *
from augmentation import *
from dataloader import LoadCocoDataset
from torch.utils.data import DataLoader

def train(
    train_loader, 
    val_loader, 
    model, 
    criterion, 
    optimizer, 
    num_epochs, 
    device, 
    checkpoint_path,
    load_model):
    
    if load_model == True:
        assert os.path.exists(checkpoint_path)
        # load the pre-trained model
        print("Loading pre-trained model ...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        history = checkpoint['history']
        prev_loss = history["val_loss"][-1]
        print("... Model successfully loaded.")
    else:
        epoch_start = 1
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
            }
        num_val_batches = len(val_loader.dataset)
        prev_loss = torch.inf

    for epoch in range(epoch_start, num_epochs+1):

        train_epoch_loss, val_epoch_loss = 0, 0
        train_epoch_acc, val_epoch_acc = 0, 0

        # training loop
        model.train()
        with tqdm.tqdm(train_loader, unit="batch") as tepoch:
            for batch, data in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                x, y = data
                x, y = x.to(device), y.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                pred = model(x)
                loss = criterion(y, pred.softmax(dim=1))
                accuracy = get_accuracy(y, pred)

                # backward
                loss.backward()

                # optmize
                optimizer.step()
                
                # print statistics
                train_epoch_loss += loss.detach().item()
                train_epoch_acc += accuracy.detach().item()
                tepoch.set_postfix(
                    loss = train_epoch_loss/(batch+1), 
                    acc = train_epoch_acc/(batch+1)
                    )
        
        # update train history
        history["train_loss"].append(train_epoch_loss/(batch+1))
        history["train_acc"].append(train_epoch_acc/(batch+1))
        
        # validation loop
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                pred = model(x)
                pred = pred.softmax(dim=1)
                loss = criterion(y, pred)
                accuracy = get_accuracy(y, pred)

                val_epoch_loss += loss.item()
                val_epoch_acc += accuracy.item()

        # update validation history
        history["val_loss"].append(val_epoch_loss/num_val_batches)
        history["val_acc"].append(val_epoch_loss/num_val_batches)

        tepoch.set_postfix(
            val_loss = history["val_loss"][-1], 
            val_acc = history["val_acc"][-1]
            )

        # checkpoint
        if history["val_loss"][-1] < prev_loss:

            print("validation loss decreased from {:.4f} to {:.4f}".format(prev_loss, history["val_loss"][-1]))
            prev_loss = history["val_loss"][-1]

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
                }, checkpoint_path)

    return history

if __name__ == "__main__":

    parser = argparse.ArgumentParser('ViT training script for mushrooms image classification', add_help=False)

    ## training parameters
    parser.add_argument('-tj', '--train_json', default="./annotations/train.json", type=str, help='train json file location')
    parser.add_argument('-vj', '--val_json', default="./annotations/val.json", type=str, help='val json file location')
    parser.add_argument('-g', '--gpu', default=0, type=int, help='GPU position')
    parser.add_argument('-is', '--image_shape', default=(224, 224), type=tuple, help='new image shape')
    parser.add_argument('-bs', '--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('-nw', '--num_workers', default=32, type=int, help='num workers') 
    parser.add_argument('-p', '--pretrained', default=True, type=bool, help='load pretrained ViT')
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-wd', '--weight_decay', default=0.1, type=float, help='weight decay')
    parser.add_argument('-ne', '--num_epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('-cp', '--checkpoint_path', default="./model/model.pt", type=str, help='checkpoint path')
    parser.add_argument('-lm', '--load_model', default=False, type=bool, help='load pre-trained model from prevoius training')

    args = parser.parse_args()
    train_path = args.train_json
    val_path = args.val_json
    gpu = args.gpu
    image_shape = args.image_shape
    batch_size = args.batch_size
    num_workers = args.num_workers
    pretrained = args.pretrained
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs
    checkpoint_path = args.checkpoint_path
    load_model = args.load_model

    # gpu or cpu
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}.")

    # loading data
    train_transforms =  T.Compose([
            T.Resize(image_shape),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(p=0.5),
            RandomAdjustGamma(0.2, p=0.5),
            RandomRotation(60, p=50),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    val_transforms =  T.Compose([
            T.Resize(image_shape),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    train_ds = LoadCocoDataset(train_path, train_transforms)
    val_ds = LoadCocoDataset(val_path, val_transforms)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # build model
    num_classes = train_ds.num_classes
    model = vit_b_16(dropout=0.2, pretrained=pretrained)
    model.heads.head = nn.Linear(in_features=768, out_features=num_classes, bias=True)

    # loss funciton
    class_weights = len(train_ds)/(num_classes * get_hist(train_ds))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32)
        )

    # optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay)

    # train
    history = train(
        train_dl, 
        val_dl, 
        model, 
        criterion, 
        optimizer, 
        num_epochs, 
        device, 
        checkpoint_path,
        load_model)

    plot_history(history)