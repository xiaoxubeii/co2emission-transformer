import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import numpy as np
import keras_nlp
from keras import ops
import include.loss as loss

# Model params.
NUM_LAYERS = 3
INTERMEDIATE_DIM = 512
NUM_HEADS = 4
DROPOUT = 0.1
NORM_EPSILON = 1e-5


@keras.saving.register_keras_serializable()
class EmissionPredictor(keras.Model):
    def __init__(self, embedd_quanti_model, bottom_layers=None, **kwargs):
        super().__init__(**kwargs)
        self.embedd_quanti_model = embedd_quanti_model
        self.quantifier = self.get_quantifying_model(embedd_quanti_model)
        self.bottom_layers = bottom_layers
        self.transf = EmissionTransformer(
            self.get_embedding_layer(self.embedd_quanti_model), None)
        self.mape_metric = keras.metrics.MeanAbsolutePercentageError()
        self.mse_metric = keras.metrics.MeanSquaredError()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse_loss = keras.losses.MeanSquaredError()
        self.mae_loss = keras.losses.MeanAbsoluteError()
        self.flattern = keras.layers.Flatten()

    def get_quantifying_model(self, model):
        # return keras.Sequential(self.embedd_quanti_model.layers[-3:])
        # return model.quantifier
        return keras.layers.Dense(1)

    def get_embedding_layer(self, model):
        # patch_layer = self.embedd_quanti_model.get_layer("patches")
        # patch_encoder = self.embedd_quanti_model.get_layer("patch_encoder")
        # patch_encoder.downstream = True
        # encoder = self.embedd_quanti_model.get_layer("mae_encoder")
        # patch_layer = model.get_layer("patches")
        # patch_encoder = model.get_layer("patch_encoder")
        # encoder = model.get_layer("mae_encoder")
        # model.embedding_layer.trainable = False
        return model.embedding_layer
        # return patch_layer, patch_encoder, encoder

    def build(self, input_shape):
        self.transf.build(input_shape)
        inputs = keras.Input(shape=input_shape)
        x = self.transf(inputs)
        x = self.flattern(x)
        self.quantifier(x)

    def call(self, inputs):
        if self.bottom_layers is not None:
            inputs = self.bottom_layers(inputs)
        x = self.transf(inputs)
        x = self.flattern(x)
        return self.quantifier(x)

    def calculate_loss(self, inputs):
        inputs, y1, y2 = inputs
        if self.bottom_layers is not None:
            inputs = self.bottom_layers(inputs)
        # y1, embedd_pred, _ = self.transf.calculate_loss((inputs, y1))
        # self.mse_metric.update_state(y1, embedd_pred)
        embedd_pred = self.transf(inputs)
        embedd_pred = self.flattern(embedd_pred)
        outputs = self.quantifier(embedd_pred)
        total_loss = self.mae_loss(y2, outputs)
        self.loss_tracker.update_state(total_loss)
        self.mape_metric.update_state(y2, outputs)
        return total_loss

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            total_loss = self.calculate_loss(inputs)

        # Apply gradients.
        train_vars = [
            self.quantifier.trainable_variables,
            self.transf.trainable_variables,
        ]
        grads = tape.gradient(total_loss, train_vars)
        tv_list = []
        for (grad, var) in zip(grads, train_vars):
            for g, v in zip(grad, var):
                tv_list.append((g, v))
        self.optimizer.apply_gradients(tv_list)
        # return {"loss": self.loss_tracker.result(), "mse": self.mse_metric.result(), "mape": self.mape_metric.result()}
        return {"loss": self.loss_tracker.result(), "mape": self.mape_metric.result()}

    def test_step(self, inputs):
        self.calculate_loss(inputs)
        return {"loss": self.loss_tracker.result(), "mape": self.mape_metric.result()}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedd_quanti_model": keras.saving.serialize_keras_object(self.embedd_quanti_model),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        for k in ["embedd_quanti_model"]:
            config[k] = keras.saving.deserialize_keras_object(config[k])
        return cls(**config)

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.mape_metric]


# @keras.saving.register_keras_serializable()
# class EmissTransformer(keras.Model):
#     def __init__(self, embedding_layer, embed_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.embed_dim = embed_dim
#         self.embedding_layer = embedding_layer

#     def build(self, input_shape):
#         self.transformer_encoder = keras_nlp.layers.TransformerEncoder(
#             intermediate_dim=INTERMEDIATE_DIM,
#             num_heads=NUM_HEADS,
#             dropout=DROPOUT,
#             layer_norm_epsilon=NORM_EPSILON,
#         )
#         self.dense = keras.layers.Dense(self.embed_dim)
#         self.dropout = keras.layers.Dropout(DROPOUT)
#         self.layernorm = keras.layers.LayerNormalization(epsilon=NORM_EPSILON)
#         self.flatten = keras.layers.Flatten()

#     def embedding(self, inputs):
#         embedding = tf.map_fn(lambda x: self.do_embedding(x), inputs)
#         positional_encoding = keras_nlp.layers.SinePositionEncoding()(embedding)
#         return embedding + positional_encoding

#     def call(self, inputs):
#         input_shape = ops.shape(inputs)
#         batch_size = input_shape[0]
#         seq_len = input_shape[1]
#         mask = compute_mask(batch_size, seq_len, seq_len, "bool")
#         out1 = inputs
#         for i in range(NUM_LAYERS):
#             out1 = self.transformer_encoder(
#                 out1, attention_mask=mask)
#         out1 = self.flatten(out1)
#         out2 = self.dense(out1)
#         out2 = self.dropout(out2)
#         return self.layernorm(out2)


def compute_mask(batch_size, n_dest, n_src, dtype):
    i = ops.arange(n_dest)[:, None]
    j = ops.arange(n_src)
    m = i >= j - n_src + n_dest
    mask = ops.cast(m, dtype)
    mask = ops.reshape(mask, [1, n_dest, n_src])
    mult = ops.concatenate(
        [ops.expand_dims(batch_size, -1), ops.convert_to_tensor([1, 1])], 0
    )
    return ops.tile(mask, mult)


def emission_transf(input_shape, embedding, bottom_layers=None):
    # inputs = keras.Input(shape=input_shape)
    # embedding.patch_encoder.downstream = True
    # transformer_encoder = keras_nlp.layers.TransformerEncoder(
    #     intermediate_dim=INTERMEDIATE_DIM,
    #     num_heads=NUM_HEADS,
    #     dropout=DROPOUT,
    #     layer_norm_epsilon=NORM_EPSILON,
    # )

    # embedding_inputs = Embedding(embedding)(inputs)
    # x = embedding_inputs
    # mask = compute_mask(batch_size, window_length, window_length, "bool")
    # for i in range(NUM_LAYERS):
    #     x = transformer_encoder(
    #         x, attention_mask=mask)

    # x = keras.layers.Flatten()(x)
    # x = keras.layers.Dense(embedding_inputs.shape[-1])(x)
    # x = keras.layers.Dropout(DROPOUT)(x)
    # outputs = keras.layers.LayerNormalization(epsilon=NORM_EPSILON)(x)
    # return keras.Model(inputs, outputs)
    et = EmissionTransformer(embedding, bottom_layers)
    et.build(input_shape)
    return et


@keras.saving.register_keras_serializable()
class EmissionTransformer(keras.Model):
    def __init__(self, embedding_model,  bottom_layers=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_model = embedding_model
        self.embedding_layer = Embedding(self.embedding_model)
        self.flatten = keras.layers.Flatten()
        self.mse = keras.losses.MeanSquaredError()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.bottom_layers = bottom_layers

    def build(self, input_shape):
        inputs = keras.Input(shape=input_shape)
        if self.bottom_layers is not None:
            inputs = self.bottom_layers(inputs)
        x = self.embedding_layer(inputs)
        self.transformer_decoder = keras_nlp.layers.TransformerDecoder(
            intermediate_dim=x.shape[-1],
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            layer_norm_epsilon=NORM_EPSILON,
        )
        x = self.transformer_decoder(x)
        # self.linear = keras.layers.Dense(x.shape[-1])
        # x = self.flatten(x)
        # self.linear(x)

    def call(self, inputs):
        if self.bottom_layers is not None:
            inputs = self.bottom_layers(inputs)
        outputs = self.embedding_layer(inputs)
        for i in range(NUM_LAYERS):
            outputs = self.transformer_decoder(outputs)
        # x = self.flatten(x)
        # return self.linear(x)
        return outputs

    def calculate_loss(self, inputs):
        x, y = inputs
        y = tf.expand_dims(y, axis=1)
        y = self.embedding_layer(y)
        y = tf.squeeze(y, axis=1)
        pred = self.call(x)
        return y, pred, self.mse(y, pred)

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            total_loss = self.calculate_loss(inputs)

        # Apply gradients.
        train_vars = [
            self.trainable_variables,
        ]
        grads = tape.gradient(total_loss, train_vars)
        tv_list = []
        for (grad, var) in zip(grads, train_vars):
            for g, v in zip(grad, var):
                tv_list.append((g, v))
        self.optimizer.apply_gradients(tv_list)
        self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, inputs):
        total_loss = self.calculate_loss(inputs)
        self.loss_tracker.update_state(total_loss)
        return {"loss": self.loss_tracker.result()}

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_model": keras.saving.serialize_keras_object(self.embedding_model),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        for k in ["embedding_model"]:
            config[k] = keras.saving.deserialize_keras_object(config[k])
        return cls(**config)

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker]


@keras.saving.register_keras_serializable()
class Embedding(keras.layers.Layer):
    def __init__(self, embedding_model, **kwargs):
        super().__init__(**kwargs)
        # self.patch_layer = patch_layer
        # self.patch_encoder = patch_encoder
        # self.encoder = encoder
        self.embedding_model = embedding_model
        self.position_encoding = keras_nlp.layers.SinePositionEncoding()
        self.flatten = keras.layers.Flatten()

    def _do_embedding(self, inputs):
        # x = self.patch_layer(inputs)
        # x = self.patch_encoder(x)
        # x = self.encoder(x)
        x = self.embedding_model(inputs)
        outputs = self.flatten(x)
        return outputs

    def call(self, input):
        embedding = tf.map_fn(lambda x: self._do_embedding(x), input)
        positional_encoding = self.position_encoding(embedding)
        return embedding + positional_encoding

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_model": keras.saving.serialize_keras_object(self.embedding_model),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        for k in ["embedding_model"]:
            config[k] = keras.saving.deserialize_keras_object(config[k])
        return cls(**config)


def emission_predictor(input_shape, emd_quant_model, bottom_layers):
    predictor = EmissionPredictor(emd_quant_model, bottom_layers)
    predictor.build(input_shape)
    return predictor
