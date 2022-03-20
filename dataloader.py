import tensorflow as tf
import json

def load_img(file_name):
    X = tf.io.decode_jpeg(
        tf.io.read_file(file_name),
        channels = 3
    )

    return tf.cast(X, tf.float32)


def load_dataset(path):
    json_file = json.load(open(path))
    num_classes = len(json_file["categories"])
    files, labels = [], []

    for img in json_file["images"]:
        files.append(img["file_name"])
        img_id = img["id"]

        for anno in json_file["annotations"]:
            if img_id == anno["image_id"]:
                labels.append(anno["category_id"])

    
    ds1 = tf.data.Dataset.from_tensor_slices(files)
    ds1 = ds1.map(lambda x: load_img(x))

    ds2 = tf.data.Dataset.from_tensor_slices(labels)
    ds2 = ds2.map(lambda x: tf.one_hot(x, num_classes))

    return tf.data.Dataset.zip((ds1, ds2))
