# Mushrooms image classification
Mushroom image classification with [Vision Transformer](https://arxiv.org/abs/2010.11929).

The dataset is taken from [Fungi Classification FGVC5 competition](https://www.kaggle.com/c/fungi-challenge-fgvc-2018), workshop at CVPR 2018.

To train the model type on terminal:
    python train.py 

where you can add these inputs:
- "-tj" or "--train_json": train json file location (default: "./annotations/train.json")
- "-vj" or "--val_json": val json file location (default: "./annotations/val.json")
- "-g" or "--gpu": GPU position (default: 0)
- "-is" or "--image_shape": new image shape (default: (224, 224))
- "-bs" or "--batch_size": batch size (default: 32)
- "-p" or "--pretrained": load pretrained ViT (default=True) 
- "-lr" or "--learning_rate": learning rate (default: 0.001)
- "-wd" or "--weight_decay": weight decay (default: 0.1)
- "-ne" or "--num_epochs": number of epochs (default: 100)
- "-cp" or "--checkpoint_path": checkpoint path (default: "./model/model.pt")
- "-lm" or "--load_model": load pre-trained model from prevoius training (default: False)

This script will save your model in the checkpoint path. Moreover, it saves the training and validation loss and accuray plot in the history.png file.

On the other hand, to evaluate the model type on terminal:
    python evaluate.py 

where you can add these inputs:
- "-tj" or "--train_json": train json file location (default: "./annotations/test.json")
- "-g" or "--gpu": GPU position (default: 0)
- "-is" or "--image_shape": new image shape (default: (224, 224))
- "-cp" or "--checkpoint_path": checkpoint path (default: "./model/model.pt")