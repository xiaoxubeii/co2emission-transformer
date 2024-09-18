# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2023
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf


@dataclass
class Generator:
    """Generator with option to augment both images and labels."""

    model_purpose: str
    batch_size: int = 32
    rotation_range: int = 0
    shift_range: float = 0
    flip: bool = False
    shear_range: float = 0
    zoom_range: float = 0
    shuffle: bool = False

    def __post_init__(self):
        self.createDataGenerator()

    def createDataGenerator(self):
        """Create data generator."""

        data_gen_args = dict(
            rotation_range=self.rotation_range,
            width_shift_range=self.shift_range,
            height_shift_range=self.shift_range,
            horizontal_flip=self.flip,
            vertical_flip=self.flip,
            shear_range=self.shear_range,
            zoom_range=self.zoom_range,
        )

        self.image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            **data_gen_args
        )

        if self.model_purpose.startswith("segmentation"):
            self.mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                **data_gen_args
            )

    def flow(self, x_data, y_data):
        """
        Flow on x (img) and y (label) data to generate:
        - segmentation: augmented images and augmented corresponding labels
        - regression: augmented images and non-augmented corresponding labels
        (emissions rate kept unchanged).
        """

        seed = 27

        if self.model_purpose.startswith("segmentation"):
            self.image_generator = self.image_datagen.flow(
                x_data, seed=seed, batch_size=self.batch_size, shuffle=self.shuffle
            )
            self.mask_generator = self.mask_datagen.flow(
                y_data, seed=seed, batch_size=self.batch_size, shuffle=self.shuffle
            )

            self.train_generator = zip(
                self.image_generator, self.mask_generator)

        elif self.model_purpose.startswith("inversion"):
            self.train_generator = self.image_datagen.flow(
                x_data,
                y_data,
                seed=seed,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
            )
        elif self.model_purpose == "embedding":
            self.train_generator = self.image_datagen.flow(
                x_data,
                y_data,
                seed=seed,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
            )
        else:
            print("Unknown model purpose in Generator")

        return self.train_generator

    def next(self):
        return self.image_generator.next(), self.mask_generator.next()

    def get_valid_data(self):
        pass


@dataclass
class ScaleDataGen(tf.keras.utils.Sequence):
    """
    Custom generator to produce pairs of background+s*plume and s*emissions for inversion.
    s is uniform random and generated between 0.5 and 4.
    """

    x: np.ndarray
    x_indexes: np.ndarray
    valid_x: np.ndarray
    valid_x_indexes: np.ndarray
    plume: np.ndarray
    xco2_back: np.ndarray
    xco2_alt_anthro: np.ndarray
    y: np.ndarray
    y_indexes: np.ndarray
    valid_y: np.ndarray
    valid_y_indexes: np.ndarray
    chans_for_scale: list
    input_size: tuple
    batch_size: int = 32
    plume_scaling_min: float = 0.25
    plume_scaling_max: float = 2
    window_length: int = 0
    scale_y: bool = True
    shuffle: bool = True

    def __post_init__(self):
        self.N_data = self.x_indexes.shape[0]
        self.N_data = self.x_indexes.shape[0]
        self.list_idx = np.arange(self.N_data)
        self.list_idx_back = np.arange(self.N_data)
        self.list_idx_alt = np.arange(self.N_data)

    def on_epoch_end(self):
        """Shuffle data at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.list_idx)
            np.random.shuffle(self.list_idx_back)
            np.random.shuffle(self.list_idx_alt)

    def __get_input(
        self,
        batches: list,
        batches_back: list,
        batches_alt: list,
        plume_scaling: np.ndarray,
        back_scaling: np.ndarray,
        alt_anthro_scaling: np.ndarray,
    ):
        """Get input batches with random scaling."""
        x_batch = self.x[self.x_indexes[batches]]

        for idx, chan in enumerate(self.chans_for_scale):
            if chan:
                if self.window_length > 0:
                    x_batch[:, :, :, :, idx: idx + 1] += (
                        plume_scaling.reshape(
                            plume_scaling.shape + (1,) * 4)
                        * self.plume[idx][self.x_indexes[batches]]
                    )

                    x_batch[:, :, :, :, idx: idx + 1] += (
                        back_scaling.reshape(
                            back_scaling.shape + (1,) * 4)
                        - self.xco2_back[idx][self.x_indexes[batches]]
                        + self.xco2_back[idx][self.x_indexes[batches_back]]
                    )

                    x_batch[:, :, :, :, idx: idx + 1] += (
                        alt_anthro_scaling.reshape(
                            alt_anthro_scaling.shape + (1,) * 4)
                        * self.xco2_alt_anthro[idx][self.x_indexes[batches_alt]]
                        - self.xco2_alt_anthro[idx][self.x_indexes[batches]]
                    )
                else:
                    x_batch[:, :, :, idx: idx + 1] += (
                        plume_scaling.reshape(plume_scaling.shape + (1,) * 3)
                        * self.plume[idx][batches]
                    )

                    x_batch[:, :, :, idx: idx + 1] += (
                        back_scaling.reshape(back_scaling.shape + (1,) * 3)
                        - self.xco2_back[idx][batches]
                        + self.xco2_back[idx][batches_back]
                    )

                    x_batch[:, :, :, idx: idx + 1] += (
                        alt_anthro_scaling.reshape(
                            alt_anthro_scaling.shape + (1,) * 3)
                        * self.xco2_alt_anthro[idx][batches_alt]
                        - self.xco2_alt_anthro[idx][batches]
                    )

        """
        self.plume_scaling = plume_scaling
        self.back_scaling = back_scaling
        self.alt_anthro_scaling = alt_anthro_scaling
        self.batches = batches
        self.batches_back = batches_back
        self.batches_alt = batches_alt
        """
        return x_batch

    def __get_output(self, batches: list, plume_scaling: np.ndarray):
        """Get output batches with random scaling."""
        y_batch = self.y[self.y_indexes[batches]]
        if self.scale_y:
            y_batch = (
                y_batch
                + plume_scaling.reshape(plume_scaling.shape +
                                        (1,) * 1) * y_batch
            )
        return y_batch

    def __get_data(self, batches: list, batches_back: list, batches_alt: list):
        """Get random batches, drawing random scaling."""
        plume_scaling = np.random.uniform(
            self.plume_scaling_min - 1, self.plume_scaling_max, size=self.batch_size
        )
        back_scaling = np.random.uniform(-3.5, 3.5, size=self.batch_size)
        alt_anthro_scaling = np.random.uniform(0.33, 3, size=self.batch_size)
        x_batch = self.__get_input(
            batches,
            batches_back,
            batches_alt,
            plume_scaling,
            back_scaling,
            alt_anthro_scaling,
        )
        y_batch = self.__get_output(batches, plume_scaling)
        return x_batch, y_batch

    def __getitem__(self, index: int):
        """Get random list of batches to draw data."""
        batches = self.list_idx[
            range(index * self.batch_size, (index + 1) * self.batch_size)
        ]
        batches_back = self.list_idx_back[
            range(index * self.batch_size, (index + 1) * self.batch_size)
        ]
        batches_alt = self.list_idx_alt[
            range(index * self.batch_size, (index + 1) * self.batch_size)
        ]
        x, y = self.__get_data(batches, batches_back, batches_alt)
        with tf.device('/cpu:0'):
            x = tf.convert_to_tensor(x, np.float64)
            y = tf.convert_to_tensor(y, np.float64)
        return x, y

    def __len__(self):
        """Get number of batches per epoch."""
        return self.N_data // self.batch_size

    def get_valid_data(self):
        return (self.valid_x[self.valid_x_indexes], self.valid_y[self.valid_y_indexes])


@dataclass
class ScaleDataGenTransformer(tf.keras.utils.Sequence):
    x: np.ndarray
    x_indexes: np.ndarray
    valid_x: np.ndarray
    valid_x_indexes: np.ndarray
    plume: np.ndarray
    xco2_back: np.ndarray
    xco2_alt_anthro: np.ndarray
    y: np.ndarray
    y_indexes: np.ndarray
    valid_y: np.ndarray
    valid_y_indexes: np.ndarray
    chans_for_scale: list
    input_size: tuple
    batch_size: int = 32
    plume_scaling_min: float = 0.25
    plume_scaling_max: float = 2
    window_length: int = 0
    scale_y: bool = True
    shuffle: bool = True

    def __post_init__(self):
        self.N_data = self.x_indexes.shape[0]
        self.list_idx = np.arange(self.N_data)
        self.list_idx_back = np.arange(self.N_data)
        self.list_idx_alt = np.arange(self.N_data)

    def on_epoch_end(self):
        """Shuffle data at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.list_idx)
            np.random.shuffle(self.list_idx_back)
            np.random.shuffle(self.list_idx_alt)

    def __get_input(
        self,
        batches: list,
        batches_back: list,
        batches_alt: list,
        plume_scaling: np.ndarray,
        back_scaling: np.ndarray,
        alt_anthro_scaling: np.ndarray,
    ):
        """Get input batches with random scaling."""
        # x_batch = np.empty(shape=(self.batch_size,) + tuple(self.input_size))
        x_batch = self.x[self.x_indexes[batches]]
        for idx, chan in enumerate(self.chans_for_scale):
            if chan:
                if self.window_length > 0:
                    x_batch[:, :, :, :, idx: idx + 1] += (
                        plume_scaling.reshape(
                            plume_scaling.shape + (1,) * 4)
                        * self.plume[idx][self.x_indexes[batches]]
                    )

                    x_batch[:, :, :, :, idx: idx + 1] += (
                        back_scaling.reshape(
                            back_scaling.shape + (1,) * 4)
                        - self.xco2_back[idx][self.x_indexes[batches]]
                        + self.xco2_back[idx][self.x_indexes[batches_back]]
                    )

                    x_batch[:, :, :, :, idx: idx + 1] += (
                        alt_anthro_scaling.reshape(
                            alt_anthro_scaling.shape + (1,) * 4)
                        * self.xco2_alt_anthro[idx][self.x_indexes[batches_alt]]
                        - self.xco2_alt_anthro[idx][self.x_indexes[batches]]
                    )
                else:
                    x_batch[:, :, :, idx: idx + 1] += (
                        plume_scaling.reshape(plume_scaling.shape + (1,) * 3)
                        * self.plume[idx][batches]
                    )

                    x_batch[:, :, :, idx: idx + 1] += (
                        back_scaling.reshape(back_scaling.shape + (1,) * 3)
                        - self.xco2_back[idx][batches]
                        + self.xco2_back[idx][batches_back]
                    )

                    x_batch[:, :, :, idx: idx + 1] += (
                        alt_anthro_scaling.reshape(
                            alt_anthro_scaling.shape + (1,) * 3)
                        * self.xco2_alt_anthro[idx][batches_alt]
                        - self.xco2_alt_anthro[idx][batches]
                    )

        return x_batch

    def __get_output(self, batches: list, plume_scaling: np.ndarray):
        """Get output batches with random scaling."""
        y_batch_0 = self.x[self.y_indexes[batches]]
        y_batch_1 = self.y[self.y_indexes[batches]]
        if self.scale_y:
            y_batch_1 = (
                y_batch_1
                + plume_scaling.reshape(plume_scaling.shape +
                                        (1,) * 1) * y_batch_1
            )
        return y_batch_0, y_batch_1

    def __getitem__(self, index: int):
        """Get random list of batches to draw data."""
        batches = self.list_idx[
            range(index * self.batch_size, (index + 1) * self.batch_size)
        ]
        batches_back = self.list_idx_back[
            range(index * self.batch_size, (index + 1) * self.batch_size)
        ]
        batches_alt = self.list_idx_alt[
            range(index * self.batch_size, (index + 1) * self.batch_size)
        ]
        x, y0, y1 = self.__get_data(batches, batches_back, batches_alt)
        # with tf.device('/cpu:0'):
        x = tf.convert_to_tensor(x, np.float64)
        y0 = tf.convert_to_tensor(y0, np.float64)
        y1 = tf.convert_to_tensor(y1, np.float64)
        return x, y0, y1

    def __get_data(self, batches: list, batches_back: list, batches_alt: list):
        """Get random batches, drawing random scaling."""
        plume_scaling = np.random.uniform(
            self.plume_scaling_min - 1, self.plume_scaling_max, size=self.batch_size
        )
        back_scaling = np.random.uniform(-3.5, 3.5, size=self.batch_size)
        alt_anthro_scaling = np.random.uniform(0.33, 3, size=self.batch_size)
        x_batch = self.__get_input(
            batches,
            batches_back,
            batches_alt,
            plume_scaling,
            back_scaling,
            alt_anthro_scaling,
        )
        y_0, y_1 = self.__get_output(batches, plume_scaling)
        return x_batch, y_0, y_1

    def __len__(self):
        """Get number of batches per epoch."""
        return self.N_data // self.batch_size

    def get_valid_data(self):
        return (self.valid_x[self.valid_x_indexes], self.valid_x[self.valid_y_indexes], self.valid_y[self.valid_y_indexes])
