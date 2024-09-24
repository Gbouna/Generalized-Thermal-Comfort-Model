import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np

class CCTTokenizer(layers.Layer):
    def __init__(self, kernel_size=3, stride=1, padding=1, pooling_kernel_size=3, pooling_stride=2,
                 num_conv_layers=2, num_output_channels=[64, 128], positional_emb=True, **kwargs):
        super().__init__(**kwargs)
        self.conv_model = keras.Sequential()
        for i in range(num_conv_layers):
            self.conv_model.add(layers.Conv2D(num_output_channels[i], kernel_size, stride, padding="valid",
                                              use_bias=False, activation="relu", kernel_initializer="he_normal"))
            self.conv_model.add(layers.ZeroPadding2D(padding))
            self.conv_model.add(layers.MaxPool2D(pooling_kernel_size, pooling_stride, "same"))
        self.positional_emb = positional_emb

    def call(self, images):
        outputs = self.conv_model(images)
        reshaped = tf.reshape(outputs, (-1, tf.shape(outputs)[1] * tf.shape(outputs)[2], tf.shape(outputs)[-1]))
        return reshaped

    def positional_embedding(self, image_size):
        if self.positional_emb:
            dummy_inputs = tf.ones((1, image_size, image_size, 3))
            dummy_outputs = self.call(dummy_inputs)
            sequence_length = tf.shape(dummy_outputs)[1]
            projection_dim = tf.shape(dummy_outputs)[-1]
            embed_layer = layers.Embedding(input_dim=sequence_length, output_dim=projection_dim)
            return embed_layer, sequence_length
        else:
            return None

class StochasticDepth(layers.Layer):
    def __init__(self, drop_prob, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prob

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (tf.shape(x)[0],) + (1,) * (tf.shape(x).shape[0] - 1)
            random_tensor = keep_prob + tf.random.uniform(shape, 0, 1)
            random_tensor = tf.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def create_cct_model(image_size=224, input_shape=(224, 224, 3), num_heads=2, projection_dim=128,
                     transformer_units=[128, 128], transformer_layers=2, stochastic_depth_rate=0.1,
                     num_classes=3, conv_layers=2, positional_emb=True):
    inputs = layers.Input(input_shape)
    cct_tokenizer = CCTTokenizer(num_conv_layers=conv_layers, positional_emb=positional_emb)
    encoded_patches = cct_tokenizer(inputs)

    pos_embed, seq_length = cct_tokenizer.positional_embedding(image_size)
    if pos_embed is not None:
        positions = tf.range(start=0, limit=seq_length, delta=1)
        position_embeddings = pos_embed(positions)
        encoded_patches += position_embeddings

    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    for i in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-5)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        x3 = StochasticDepth(dpr[i])(x3)

        # Ensure x3 has the same number of channels as x2
        if x3.shape[-1] != x2.shape[-1]:
            x3 = layers.Dense(x2.shape[-1])(x3)

        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
    attention_weights = tf.nn.softmax(layers.Dense(1)(representation), axis=1)
    weighted_representation = tf.matmul(attention_weights, representation, transpose_a=True)
    weighted_representation = tf.squeeze(weighted_representation, -2)
    logits = layers.Dense(num_classes)(weighted_representation)
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def get_cct_model():
    model = create_cct_model(
        image_size=224,
        input_shape=(224, 224, 3),
        num_heads=2,
        projection_dim=128,
        transformer_units=[128, 128],
        transformer_layers=2,
        stochastic_depth_rate=0.1,
        num_classes=3,
        conv_layers=2,
        positional_emb=True
    )
    model.compile(
        optimizer=tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.0001),
        loss=keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy"),
                 keras.metrics.TopKCategoricalAccuracy(5, name="top-5-accuracy")]
    )
    return model