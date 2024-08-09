from tensorflow import random_normal_initializer

from keras import Model as KerasModel
from keras.models import Sequential
from keras.layers import Conv2DTranspose, BatchNormalization, ReLU, Concatenate, Input
from keras.losses import SparseCategoricalCrossentropy
from keras.applications.mobilenet_v2 import MobileNetV2


class UpSampler(Sequential):  # type: ignore[misc]

    def __init__(self, filters: int, size: int):
        super(UpSampler, self).__init__()

        self.add(
            Conv2DTranspose(
                filters,
                size,
                strides=2,
                padding='same',
                kernel_initializer=random_normal_initializer(0., 0.02),
                use_bias=False
            )
        )

        self.add(BatchNormalization())
        self.add(ReLU())


def UNetModel(classCount: int, imageSize: int) -> KerasModel:
    def buildDownStack() -> KerasModel:
        baseModel: KerasModel = MobileNetV2(
            input_shape=(imageSize, imageSize, 3),
            include_top=False
        )

        layerNames = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]

        baseModelOutputs = [baseModel.get_layer(name).output for name in layerNames]

        downStack = KerasModel(
            inputs=baseModel.input,
            outputs=baseModelOutputs
        )
        downStack.trainable = False

        return downStack

    def buildUpStack() -> list[UpSampler]:
        return [
            UpSampler(512, 3),
            UpSampler(256, 3),
            UpSampler(128, 3),
            UpSampler(64, 3),
        ]

    def unet_model(output_channels: int) -> KerasModel:
        inputs = Input(shape=[imageSize, imageSize, 3])

        downStack = buildDownStack()
        upStack = buildUpStack()

        # Downsampling through the model
        skips = downStack(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(upStack, skips):
            x = up(x)
            concat = Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = Conv2DTranspose(
            filters=output_channels,
            kernel_size=3,
            strides=2,
            padding='same'
        )  # 64x64 -> 128x128

        x = last(x)
        return KerasModel(inputs=inputs, outputs=x)

    model = unet_model(output_channels=classCount)
    model.compile(
        optimizer='adam',
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model
