import argparse
import os
import tqdm

import torch
import torchvision.transforms as T
from torchvision.models import VisionTransformer

from augmentation import *
from dataloader import LoadCocoDataset

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
    assert os.path.exists(args.checkpoint_path)

    print("Loading pre-trained model ...")
    checkpoint = torch.load(args.checkpoint_path)
    model = VisionTransformer(
        image_size = checkpoint["image_shape"],
        patch_size = checkpoint["patch_size"],
        num_layers = checkpoint["num_layers"],
        num_heads = checkpoint["num_heads"],
        hidden_dim = checkpoint["hidden_dim"],
        mlp_dim = checkpoint["mlp_dim"],
        dropout = checkpoint["dropout"],
        attention_dropout = checkpoint["attention_dropout"],
        num_classes = checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    print("... Model successfully loaded.")
    
    # evaluation
    accuracy = evaluate(val_ds, model, device)
    
    print("Accuracy: {:.3f} %".format(accuracy*100))
