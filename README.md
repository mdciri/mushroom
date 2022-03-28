# ViT for Mushrooms Image Classification
Mushroom image classification using [Vision Transformer](https://arxiv.org/abs/2010.11929). The implementation was done using pytorch (version 1.11.0).

The dataset is taken from [Fungi Classification FGVC5 competition](https://www.kaggle.com/c/fungi-challenge-fgvc-2018), workshop at CVPR 2018. The dataset describes 1394 different classes of mushrooms and it is split into training, validation, and test. Each of these datasets contains 85578, 4182, and 9758 images respectively.


**Training**

To train the model type on terminal:

    python train.py 

where you can add these inputs:
- `-tj` or `--train_json`: train json file location (default: `./annotations/train.json`)
- `-vj` or `--val_json`: val json file location (default: `./annotations/val.json`)
- `-g` or `--gpu`: GPU position (default: 0)
- `-is` or `--image_shape`: new image shape (default: (224, 224))
- `-bs` or `--batch_size`: batch size (default: 32)
- `-nw` or `--num_workers`: num workers (default: 2) 
- `-p` or `--pretrained`: load pretrained ViT (default=True) 
- `-lr` or `--learning_rate`: learning rate (default: 0.001)
- `-wd` or `--weight_decay`: weight decay (default: 0.1)
- `-ne` or `--num_epochs`: number of epochs (default: 100)
- `-cp` or `--checkpoint_path`: checkpoint path (default: `./model/model.pt`)
- `-lm` or `--load_model`: load pre-trained model from prevoius training (default: False)

This script saves your model in the checkpoint path. Moreover, it saves the training and validation loss and accuray plot in the history.png file.


**Evaluation**

To evaluate the model type on terminal:

    python evaluate.py 

where you can add these inputs:
- `-vj` or `--val_json`: train json file location (default: `./annotations/val.json`)
- `-g` or `--gpu`: GPU position (default: 0)
- `-is` or `--image_shape`: new image shape (default: (224, 224))
- `-cp` or `--checkpoint_path`: checkpoint path (default: `./model/model.pt`)

This script calculates the accuracy of the input dataset using the pre-trained model saved in the checkpoint path.


**Testing**

To test the model type on terminal:

    python test.py 

where you can add these inputs:
- `-tj` or `--val_json`: train json file location (default: `./annotations/val.json`)
- `-cj`, or `--classes_json`: classes dictionary location (default `./classes_id_names.json`)
- `-g` or `--gpu`: GPU position (default: 0)
- `-is` or `--image_shape`: new image shape (default: (224, 224))
- `-cp` or `--checkpoint_path`: checkpoint path (default: `./model/model.pt`)

This scripts returns a .csv file which contains 2 columns: *id* and *predicitons*.

The *id* column contains the image IDs of the dataset, whereas *predicitons* the top 3 predictions of the model.