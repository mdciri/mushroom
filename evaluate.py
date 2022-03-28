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


def evaluate(test_ds, model, device):

    predictions = {
        "id": [],
        "predicted": []
    }
    
    model = model.to(device)
    running_accuracy = 0

    for id, x, y in test_ds:

        x = torch.unsqueeze(x, dim=0)
        y = torch.unsqueeze(y, dim=0)  
        x, y = x.to(device), y.to(device)
        pred = model(x)

        y = torch.argmax(y)
        pred = torch.argmax(pred.softmax(dim=1))

        running_accuracy += get_accuracy(y, pred)
        pred_top3 = torch.topk(pred, 3)
        pred_top3 = pred_top3.indices.numpy()

        predictions["id"].append(id)
        predictions["predicted"].append(f"{pred_top3[0]} {pred_top3[1]} {pred_top3[2]}")

    print("Accuracy: {:.4f}".format(running_accuracy/len(test_ds)))

    return pd.DataFrame(predictions)


if __name__ == "__main__":

    parser = argparse.ArgumentParser('ViT evaluation script for mushrooms image classification', add_help=False)

    ## evaluation parameters
    parser.add_argument('-tj', '--test_json', default="./annotations/test.json", type=str, help='test json file location')
    parser.add_argument('-g', '--gpu', default=0, type=int, help='GPU position')
    parser.add_argument('-is', '--image_shape', default=(224, 224), type=tuple, help='new image shape')
    parser.add_argument('-cp', '--checkpoint_path', default="./model/model.pt", type=str, help='checkpoint path')

    args = parser.parse_args()
    test_path = args.test_json
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
    test_ds = LoadCocoDataset(test_path, transforms, return_id=True)

    # load model
    print("Loading pre-trained model ...")
    model = vit_b_16(dropout=0.2, pretrained=False)
    model.heads.head = nn.Linear(in_features=768, out_features=test_ds.num_classes, bias=True)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("... Model successfully loaded.")

    # evaluation
    df = evaluate(test_ds, model, device)
    
    # save dataframe
    df.to_csv("test_predictions.csv")