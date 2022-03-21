import tensorflow as tf
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom

class Augmentation(tf.keras.Model):
    def __init__(self, flip_mode="horizontal_and_vertical", rotation_factor=0.2, scale_factor=0.2, brigth_factor=0.2):
        super(Augmentation, self).__init__(name='Augmentation')

        self.flip = RandomFlip(
            mode=flip_mode
        )
        self.rotate = RandomRotation(
            factor = rotation_factor, 
            fill_mode='constant', 
            fill_value = -1.
        )
        self.scale = RandomZoom(
            factor = scale_factor, 
            fill_mode='constant', 
            fill_value = -1.
        )
        self.brigth_factor = brigth_factor

    def call(self, x):

        x = self.scale(self.rotate(self.flip(x)))

        gamma = tf.random.uniform(shape=[], minval=1-self.brigth_factor, maxval=1+self.brigth_factor)
        x = (x+1)/2 # x in range [0-1]
        x = x**gamma
        x = 2*x -1 # x in range [-1, 1]

        return x
        