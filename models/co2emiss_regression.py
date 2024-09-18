import keras
import tensorflow as tf


def co2emiss_regres(input_shape, embedding, **kwargs):
    inputs = keras.Input(shape=input_shape, batch_size=32)
    # x = embedding.patch_layer(inputs)
    # x = embedding.patch_encoder(x)
    # x = embedding.encoder(x)
    # x = keras.layers.Flatten()(x)
    # x = keras.layers.Dense(128, activation='relu')(x)
    # x = keras.layers.Dense(1)(x)
    # outputs = keras.layers.LeakyReLU(alpha=0.3)(x)
    # return keras.Model(inputs, outputs, name="co2emiss_regres")
    regres = EmissRegression(embedding)
    regres.build(input_shape)
    return regres


@keras.saving.register_keras_serializable()
class EmissRegression(keras.Model):
    def __init__(self, embedding_model, bottom_layers=None, **kwargs):
        super().__init__(**kwargs)
        self.embedding_model = embedding_model
        self.embedding_layer = self.get_embedding_layer(self.embedding_model)
        self.bottom_layers = bottom_layers
        self.mae_loss = keras.losses.MeanAbsoluteError()
        self.mape_metric = keras.metrics.MeanAbsolutePercentageError()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.flattern = keras.layers.Flatten()
        self.quantifier = keras.Sequential([
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(1),
            keras.layers.LeakyReLU(alpha=0.3)
        ])

    def get_embedding_layer(self, model):
        model.patch_encoder.downstream = True
        return keras.Sequential([
            model.patch_layer,
            model.patch_encoder,
            model.encoder
        ])

    def build(self, input_shape):
        inputs = keras.Input(shape=input_shape)
        if self.bottom_layers is not None:
            inputs = self.bottom_layers(inputs)
        x = self.embedding_layer(inputs)
        x = self.flattern(x)
        self.quantifier(x)

    def call(self, inputs):
        if self.bottom_layers is not None:
            inputs = self.bottom_layers(inputs)
        x = self.embedding_layer(inputs)
        x = self.flattern(x)
        return self.quantifier(x)

    def calculate_loss(self, inputs):
        inputs, y = inputs
        if self.bottom_layers is not None:
            inputs = self.bottom_layers(inputs)
        x = self.embedding_layer(inputs)
        x = self.flattern(x)
        outputs = self.quantifier(x)
        loss = self.mae_loss(y, outputs)
        self.loss_tracker.update_state(loss)
        self.mape_metric.update_state(y, outputs)
        return loss

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            total_loss = self.calculate_loss(inputs)

         # Apply gradients.
        train_vars = [
            self.quantifier.trainable_variables,
            self.embedding_layer.trainable_variables
        ]
        grads = tape.gradient(total_loss, train_vars)
        tv_list = []
        for (grad, var) in zip(grads, train_vars):
            for g, v in zip(grad, var):
                tv_list.append((g, v))
        self.optimizer.apply_gradients(tv_list)
        return {"loss": self.loss_tracker.result(), "mape": self.mape_metric.result()}

    def test_step(self, inputs):
        self.calculate_loss(inputs)
        return {"loss": self.loss_tracker.result(), "mape": self.mape_metric.result()}

    @classmethod
    def from_config(cls, config):
        for k in ["embedding_model"]:
            config[k] = keras.saving.deserialize_keras_object(config[k])
        return cls(**config)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_model": keras.saving.serialize_keras_object(self.embedding_model),
            }
        )
        return config

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.mape_metric]
