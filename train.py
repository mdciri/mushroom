import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint

import os
import json
import argparse
from dataloader import load_dataset
from augmentation import Augmentation
from layers_and_model import ViT

def train(args):

    train_json = json.load(open(args.train_file))
    valid_json = json.load(open(args.valid_file))
    h, w = args.new_shape
    gpu = args.gpu
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    # get number of classes
    num_classes = len(train_json["categories"])

    # load training and validation dataset
    train_ds, freq = load_dataset(train_json, num_classes, "train")
    valid_ds = load_dataset(valid_json, num_classes)

    # class weights
    num_train_images = len(train_json["images"])
    weights = num_train_images / (num_classes * np.asarray(list(freq.values())))
    class_weights = {}
    for i, key in enumerate(freq.keys()):
        class_weights[key] = weights[i]

    # preprocessing
    def preprocessing(img):

        new_img = 2*img/255. -1
        new_img = tf.image.resize(new_img, [h, w], antialias=True)

        return tf.cast(new_img, tf.float32)

    train_ds = train_ds.map(lambda x, y: (preprocessing(x), y))
    valid_ds = valid_ds.map(lambda x, y: (preprocessing(x), y))

    # training params 
    learning_rate = 1e-3
    patch_size = 32
    num_patches = h*w//(patch_size**2)
    num_layers = 12
    num_heads = 12
    projection_dim = 768
    mlp_head_units = [2048, 1024]
    transformer_units = [projection_dim*2, projection_dim]
    drop_rate = 0.2

    # load model
    net = ViT(
        num_classes, 
        mlp_head_units,
        patch_size,
        num_patches,
        num_layers,
        num_heads, 
        projection_dim, 
        transformer_units, 
        drop_rate)

    optimizer = Adam(
        learning_rate=learning_rate, 
    )

    net.compile(
        optimizer = optimizer,
        loss = CategoricalCrossentropy(from_logits=False),
        metrics=[
            "acc",
        ],
    )

    checkpoint_filepath = "/tmp/checkpoint"
    callbacks = [ModelCheckpoint(
        checkpoint_filepath,
        save_best_only=True,
        save_weights_only=True,
    )]

    # data augmentation
    flip_mode="horizontal_and_vertical"
    rotation_factor = 0.2
    scale_factor = 0.2
    brigth_factor = 0.2

    aug = Augmentation(
        flip_mode,
        rotation_factor, 
        scale_factor, 
        brigth_factor
    )
    
    # training
    history = net.fit(
        x = train_ds.repeat().shuffle(num_train_images).batch(batch_size).map(lambda x, y: (aug(x), y)).prefetch(2),
        validation_data = valid_ds.batch(batch_size).prefetch(2),
        class_weight = class_weights,
        steps_per_epoch = np.ceil(num_train_images/batch_size),
        epochs = num_epochs,
        callbacks = callbacks,
    )

    return history


if __name__ == "__main__":

    parser = argparse.ArgumentParser('ViT training script', add_help=False)

    ## training parameters
    parser.add_argument('-t', '--train_file', default="./annotataions/train.json", type=str, help='json training annotations')
    parser.add_argument('-v', '--valid_file', default="./annotataions/val.json", type=str, help='json valid annotations')
    parser.add_argument('-ns', '--new_shape', default=(256, 256), type=tuple, help='new image shape')
    parser.add_argument('-g', '--gpu', default=0, type=int, help='GPU position')
    parser.add_argument('-bs', '--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('-ne', '--num_epochs', default=200, type=int, help='number of epochs')

    args = parser.parse_args()

    

    