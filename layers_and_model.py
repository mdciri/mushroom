import tensorflow as tf
from tensorflow.keras.layers import Flatten, Conv2D, Dense, Embedding, LayerNormalization, MultiHeadAttention, Dropout

class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, patch_size, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.conv = Conv2D(projection_dim, 
                           kernel_size=patch_size, 
                           strides=patch_size, 
                           padding='valid', 
                           kernel_initializer='he_normal')
        self.position_embedding = Embedding(input_dim=num_patches, 
                                            output_dim=projection_dim)

    def call(self, x):
        bs = x.shape[0]
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        out = tf.reshape(self.conv(x), (bs, -1, self.projection_dim)) + self.position_embedding(positions)
        return out

class MultiLayerPerceptron(tf.keras.layers.Layer):
    def __init__(self, hidden_units, drop_rate):
        super(MultiLayerPerceptron, self).__init__()

        self.dense = [Dense(units, 
                            activation=tf.nn.gelu) for units in hidden_units]
        self.drop = Dropout(drop_rate)
    
    def call(self, x):

        for dense_lay in self.dense:
            x = dense_lay(x)

        return self.drop(x)

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, num_heads, projection_dim, transformer_units, drop_rate):
        super(TransformerEncoder, self).__init__()

        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.mha = MultiHeadAttention(num_heads=num_heads, 
                                      key_dim=projection_dim, 
                                      dropout=drop_rate)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.mlp = MultiLayerPerceptron(hidden_units=transformer_units, 
                                        drop_rate=drop_rate)

    def call(self, x):

        x_norm = self.norm1(x)
        x2 = self.mha(x_norm, x_norm) + x
        out = self.mlp(self.norm2(x2)) + x2
        return x

class VisionTransformer(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_heads, projection_dim, transformer_units, drop_rate):
        super(VisionTransformer, self).__init__()

        self.encoders = [TransformerEncoder(num_heads, 
                                            projection_dim, 
                                            transformer_units, 
                                            drop_rate) for _ in range(num_layers)]

    def call(self, x):

        for enc_lay in self.encoders:
            x = enc_lay(x)

        return x

class Classifier(tf.keras.layers.Layer):
    def __init__(self, num_classes, mlp_head_units, drop_rate):
        super(Classifier, self).__init__()

        self.norm = LayerNormalization(epsilon=1e-6)
        self.drop = Dropout(drop_rate)
        self.flat = Flatten()
        self.mlp = MultiLayerPerceptron(hidden_units=mlp_head_units, 
                                        drop_rate=drop_rate)
        self.dense = Dense(num_classes, activation='softmax')

    def call(self, x):

        x = self.drop(self.flat(self.norm(x)))
        x = self.mlp(x)
        out = self.dense(x)

        return out

class ViT(tf.keras.Model):
    def __init__(self, 
                 num_classes, 
                 mlp_head_units,
                 patch_size,
                 num_patches,
                 num_layers,
                 num_heads, 
                 projection_dim, 
                 transformer_units, 
                 drop_rate=0.1):
        super(ViT, self).__init__()

        self.patch_encoder = PatchEncoder(patch_size, num_patches, projection_dim)
        self.transformer = VisionTransformer(num_layers, num_heads, projection_dim, transformer_units, drop_rate)
        self.classifier = Classifier(num_classes, mlp_head_units, drop_rate)

    def call(self, x):

        x = self.patch_encoder(x)
        x = self.transformer(x)
        out = self.classifier(x)

        return out