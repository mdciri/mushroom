import os
import argparse
import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import VisionTransformer
import torch.optim as optim

from utils import *
from augmentation import get_train_transforms, get_test_val_transforms
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
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_start = checkpoint["epoch"]
        history = checkpoint["history"]
        prev_loss = history["val_loss"][-1]
        print("... Model successfully loaded.")
    else:
        checkpoint = {
            "image_size": model.image_size,
            "patch_size": model.path_size,
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "hidden_dim": model.hidden_dim,
            "mlp_dim": model.mlp_dim,
            "dropout": model.dropout,
            "attention_dropout": model.attention_dropout,
            "num_classes": model.num_classes,
            "epoch": 1,
            "model_state_dict": None,
            "optimizer_state_dict": None,
            "history": None
        }
        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
            }
        prev_loss = torch.inf

    num_val_batches = len(val_loader.dataset)

    for epoch in range(checkpoint["epoch"], num_epochs+1):

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

            # update  and save checkpoint
            checkpoint["epoch"] = epoch,
            checkpoint["model_state_dict"] = model.state_dict(),
            checkpoint["optimizer_state_dict"] = optimizer.state_dict(),
            checkpoint["history"] = history

            torch.save(checkpoint, checkpoint_path)

    return checkpoint

if __name__ == "__main__":

    parser = argparse.ArgumentParser("ViT training script for mushrooms image classification", add_help=False)

    ## training parameters
    parser.add_argument("-tj", "--train_json", default="./annotations/train.json", type=str, help="train json file location")
    parser.add_argument("-vj", "--val_json", default="./annotations/val.json", type=str, help="val json file location")
    parser.add_argument("-g", "--gpu", default=0, type=int, help="GPU position")
    parser.add_argument("-is", "--image_shape", default=(224, 224), type=tuple, help="new image shape")
    parser.add_argument("-bs", "--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("-nw", "--num_workers", default=2, type=int, help="num workers") 
    parser.add_argument("-lr", "--learning_rate", default=1e-3, type=float, help="learning rate")
    parser.add_argument("-wd", "--weight_decay", default=0.1, type=float, help="weight decay")
    parser.add_argument("-ne", "--num_epochs", default=100, type=int, help="number of epochs")
    parser.add_argument("-cp", "--checkpoint_path", default="./model/model.pt", type=str, help="checkpoint path")
    parser.add_argument("-lm", "--load_model", default=False, type=bool, help="load pre-trained model from prevoius training")

    # augmentation parameters
    parser.add_argument("-pa", "--perc_augmentation", default=0.7, type=float, help="augmentation percentage")
    parser.add_argument("-phf", "--perc_horiz_filp", default=0.5, type=float, help="random horzontal flip percentage")
    parser.add_argument("-pvf", "--perc_vert_filp", default=0.5, type=float, help="random vertical flip percentage")
    parser.add_argument("-pr", "--perc_rotation", default=0.5, type=float, help="random rotation percentage")
    parser.add_argument("-rr", "--rotation_range", default=60, type=int, help="rotation range")
    parser.add_argument("-pb", "--perc_bright", default=0.5, type=float, help="random brightness percentage")
    parser.add_argument("-gr", "--gamma_range", default=0.2, type=float, help="random brightness gamma range")

    ## model parameters
    parser.add_argument("-ps", "--patch_size", default=16, type=int, help="image patch size")
    parser.add_argument("-nl", "--num_layers", default=12, type=int, help="number of encoder layers")
    parser.add_argument("-nh", "--num_heads", default=12, type=int, help="number of heads for the MHA layer")
    parser.add_argument("-hd", "--hidden_dim", default=768, type=int, help="hidden dimension")
    parser.add_argument("-md", "--mlp_dim", default=3072, type=int, help="mlp dimension")
    parser.add_argument("-d", "--dropout", default=0.2, type=float, help="dropout rate")
    parser.add_argument("-da", "--attention_dropout", default=0.2, type=float, help="attention dropout rate")

    args = parser.parse_args()

    # gpu or cpu
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}.")

    # loading data
    train_transforms =  get_train_transforms(
        args.image_shape,
        args.perc_augmentation,
        args.perc_horiz_filp,
        args.perc_vert_filp,
        args.gamma_range,
        args.perc_bright,
        args.rotation_range,
        args.perc_rotation
    )
    val_transforms =  get_test_val_transforms(args.image_shape)

    train_ds = LoadCocoDataset(args.train_json, train_transforms)
    val_ds = LoadCocoDataset(args.val_json, val_transforms)
    train_dl = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers)
    val_dl = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers)

    # build model
    num_classes = train_ds.num_classes
    model = VisionTransformer(
        image_size = args.image_shape[0],
        patch_size = args.patch_size,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        hidden_dim = args.hidden_dim,
        mlp_dim = args.mlp_dim,
        dropout = args.dropout,
        attention_dropout = args.attention_dropout,
        num_classes = num_classes,
    )

    # loss funciton
    class_weights = len(train_ds)/(num_classes * get_hist(train_ds))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32)
        )

    # optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)

    # train
    checkpoint = train(
        train_dl, 
        val_dl, 
        model, 
        criterion, 
        optimizer, 
        args.num_epochs, 
        device, 
        args.checkpoint_path,
        args.load_model)

    plot_history(checkpoint["history"])