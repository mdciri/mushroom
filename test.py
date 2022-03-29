import argparse
import os
import tqdm

import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import VisionTransformer

from augmentation import get_test_val_transforms
from dataloader import LoadCocoTestDataset

def test(test_ds, model, device):

    predictions = {
        "id": [],
        "predicted": []
    }
    
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for x, id in tqdm.tqdm(test_ds, total=len(test_ds)):

            x = torch.unsqueeze(x, dim=0)
            x = x.to(device)
            pred = model(x)
            pred_top3 = torch.topk(pred, 3)
            pred_top3 = pred_top3.indices.numpy()

            predictions["id"].append(id)
            predictions["predicted"].append(f"{pred_top3[0]} {pred_top3[1]} {pred_top3[2]}")

    return pd.DataFrame(predictions)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("ViT evaluation script for mushrooms image classification", add_help=False)

    ## testing parameters
    parser.add_argument("-tj", "--test_json", default="./annotations/test.json", type=str, help="test json file location")
    parser.add_argument("-cj", "--classes_json", default="./classes_id_names.json", type=str, help="classes dictionary")
    parser.add_argument("-g", "--gpu", default=0, type=int, help="GPU position")
    parser.add_argument("-is", "--image_shape", default=(224, 224), type=tuple, help="new image shape")
    parser.add_argument("-cp", "--checkpoint_path", default="./model/model.pt", type=str, help="checkpoint path")
    args = parser.parse_args()
    
    # gpu or cpu
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}.")

    # load data
    transforms = get_test_val_transforms(args.image_shape)
    test_ds = LoadCocoTestDataset(args.test_json, transforms)

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
    df = test(test_ds, model, device)
    
    # save dataframe
    df.to_csv("test_predictions.csv")