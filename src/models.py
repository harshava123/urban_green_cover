"""Model definitions: custom CNN and transfer learning backbones."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


def build_custom_cnn(
    input_shape: tuple[int, int, int],
    num_classes: int,
    filters: list[int],
    dropout: float = 0.4,
) -> tf.keras.Model:
    """Simple custom CNN baseline for teaching comparisons."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255.0)(inputs)

    for i, f in enumerate(filters):
        x = layers.Conv2D(f, 3, padding="same", activation="relu", name=f"conv_{i+1}")(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.MaxPooling2D(2, name=f"pool_{i+1}")(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return models.Model(inputs, outputs, name="custom_cnn")


def _get_backbone(name: str, input_shape: tuple[int, int, int]) -> tf.keras.Model:
    """Load a pretrained backbone without the classification head."""
    weights = "imagenet"
    include_top = False
    pooling = "avg"

    if name == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=include_top, weights=weights, pooling=pooling
        )
    elif name == "resnet50":
        base = tf.keras.applications.ResNet50(
            input_shape=input_shape, include_top=include_top, weights=weights, pooling=pooling
        )
    elif name == "efficientnetb0":
        base = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape, include_top=include_top, weights=weights, pooling=pooling
        )
    elif name == "densenet121":
        base = tf.keras.applications.DenseNet121(
            input_shape=input_shape, include_top=include_top, weights=weights, pooling=pooling
        )
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    base.trainable = False
    return base


class _ResNetMeanSubtract(layers.Layer):
    """ImageNet mean subtraction for ResNet50 (serializable alternative to preprocess_input)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)

    def call(self, inputs):
        return inputs - self._mean

    def get_config(self):
        return super().get_config()


def _preprocess_for_backbone(backbone_name: str, inputs: tf.Tensor) -> tf.Tensor:
    """Apply backbone-specific preprocessing to 0-255 RGB inputs."""
    if backbone_name == "mobilenetv2":
        return layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    if backbone_name == "resnet50":
        return _ResNetMeanSubtract()(inputs)
    if backbone_name in ("efficientnetb0", "densenet121"):
        return layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    return inputs


def build_transfer_model(
    backbone_name: str,
    input_shape: tuple[int, int, int],
    num_classes: int,
    dropout: float = 0.3,
    fine_tune_at: int | None = None,
) -> tf.keras.Model:
    """Transfer learning model with optional partial fine-tuning."""
    inputs = layers.Input(shape=input_shape)
    backbone = _get_backbone(backbone_name, input_shape)
    x = _preprocess_for_backbone(backbone_name, inputs)
    x = backbone(x, training=False)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = models.Model(inputs, outputs, name=backbone_name)

    if fine_tune_at is not None:
        backbone.trainable = True
        for layer in backbone.layers[:fine_tune_at]:
            layer.trainable = False

    return model


def build_model(
    model_name: str,
    config: dict,
    num_classes: int,
) -> tf.keras.Model:
    """Factory for all supported model variants."""
    image_size = config["data"]["image_size"]
    input_shape = (image_size[0], image_size[1], 3)

    if model_name == "custom_cnn":
        cnn_cfg = config["models"]["custom_cnn"]
        return build_custom_cnn(
            input_shape=input_shape,
            num_classes=num_classes,
            filters=cnn_cfg["filters"],
            dropout=cnn_cfg["dropout"],
        )

    return build_transfer_model(
        backbone_name=model_name,
        input_shape=input_shape,
        num_classes=num_classes,
    )


def model_param_count(model: tf.keras.Model) -> int:
    """Return total trainable + non-trainable parameter count."""
    return int(model.count_params())
