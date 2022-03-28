import argparse
import tqdm

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import vit_b_16

from augmentation import *
from dataloader import LoadCocoDataset
from utils import get_accuracy


def evaluate(val_ds, model, device):

    trues = []
    predictions = []

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for x, y in tqdm.tqdm(val_ds, total=len(val_ds)):

            x = torch.unsqueeze(x, dim=0)
            x = x.to(device)
            pred = model(x)
            
            y = torch.argmax(y, dim=1)
            pred = torch.argmax(pred.softmax(dim=1), dim=1)

            trues.append(y)
            predictions.append(pred)

    accuracy = (trues == predictions).type(torch.float).sum() / len(val_ds)

    return accuracy


if __name__ == "__main__":

    parser = argparse.ArgumentParser('ViT evaluation script for mushrooms image classification', add_help=False)

    ## evaluation parameters
    parser.add_argument('-vj', '--val_json', default="./annotations/val.json", type=str, help='validation json file location')
    parser.add_argument('-g', '--gpu', default=0, type=int, help='GPU position')
    parser.add_argument('-is', '--image_shape', default=(224, 224), type=tuple, help='new image shape')
    parser.add_argument('-cp', '--checkpoint_path', default="./model/model.pt", type=str, help='checkpoint path')

    args = parser.parse_args()
    val_path = args.val_json
    gpu = args.gpu
    image_shape = args.image_shape
    checkpoint_path = args.checkpoint_path
    
    # gpu or cpu
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}.")

    # load data
    transforms = T.Compose([
            T.Resize(image_shape),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    val_ds = LoadCocoDataset(val_path, transforms)

    # load model
    print("Loading pre-trained model ...")
    model = vit_b_16(dropout=0.2, pretrained=False)
    model.heads.head = nn.Linear(in_features=768, out_features=val_ds.num_classes, bias=True)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("... Model successfully loaded.")

    # evaluation
    accuracy = evaluate(val_ds, model, device)
    
    print("Accuracy: {:.3f} %".format(accuracy*100))
