import os
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import BatchNormalization, Conv2D, Softmax
from tensorflow.keras.models import Model
from ei_tensorflow.constrained_object_detection import models, dataset, metrics, util
from ei_shared.pretrained_weights import get_or_download_pretrained_weights
import ei_tensorflow.training
from pathlib import Path
import requests

# Set weights prefix
WEIGHTS_PREFIX = os.environ.get('WEIGHTS_PREFIX', os.getcwd())


def build_model(input_shape: tuple, weights: str, alpha: float, num_classes: int) -> tf.keras.Model:
    """
    Construct a constrained object detection model.

    Args:
        input_shape (tuple): Passed to MobileNet construction.
        weights (str): Weights for initialization of MobileNet where None implies random initialization.
        alpha (float): MobileNet alpha value.
        num_classes (int): Number of classes, i.e., final dimension size in output.

    Returns:
        Uncompiled keras model.
        Model takes (B, H, W, C) input and returns (B, H//8, W//8, num_classes) logits.
    """
    # Create full MobileNetV2 model
    mobile_net_v2 = MobileNetV2(input_shape=input_shape, weights=weights, alpha=alpha, include_top=True)

    # Adjust batch normalization layers for smaller networks
    for layer in mobile_net_v2.layers:
        if isinstance(layer, BatchNormalization):
            layer.momentum = 0.9

    # Cut MobileNet where it hits 1/8th input resolution (HW/8, HW/8, C)
    cut_point = mobile_net_v2.get_layer('block_6_expand_relu')

    # Attach additional head to MobileNet
    model = Conv2D(filters=32, kernel_size=1, strides=1, activation='relu', name='head')(cut_point.output)
    logits = Conv2D(filters=num_classes, kernel_size=1, strides=1, activation=None, name='logits')(model)

    return Model(inputs=mobile_net_v2.input, outputs=logits)


def train(
        num_classes: int, learning_rate: float, num_epochs: int, alpha: float, object_weight: float,
        train_dataset: tf.data.Dataset, validation_dataset: tf.data.Dataset, best_model_path: str,
        input_shape: tuple, batch_size: int, lr_finder: bool = False, ensure_determinism: bool = False
) -> tf.keras.Model:
    """
    Construct and train a constrained object detection model.

    Args:
        num_classes (int): Number of classes in datasets.
        learning_rate (float): Learning rate for Adam optimizer.
        num_epochs (int): Number of epochs passed to model.fit().
        alpha (float): Alpha used to construct MobileNet.
        object_weight (float): The weighting to give the object in the loss function.
        train_dataset (tf.data.Dataset): Training dataset.
        validation_dataset (tf.data.Dataset): Validation dataset.
        best_model_path (str): Path to save the best model.
        input_shape (tuple): Shape of the model's input.
        batch_size (int): Training batch size.
        lr_finder (bool): Use learning rate finder if True.
        ensure_determinism (bool): Ensure deterministic operations if True.

    Returns:
        Trained keras model.
    """
    num_classes_with_background = num_classes + 1
    width, height, input_num_channels = input_shape

    if width != height:
        raise ValueError(f"Only square inputs are supported; not {input_shape}")

    input_width_height = width

    # Get pretrained weights if available
    allowed_combinations = [{'num_channels': 1, 'alpha': 0.1}, {'num_channels': 1, 'alpha': 0.35},
                            {'num_channels': 3, 'alpha': 0.1}, {'num_channels': 3, 'alpha': 0.35}]

    weights = get_or_download_pretrained_weights(WEIGHTS_PREFIX, input_num_channels, alpha, allowed_combinations)

    # Build the model
    model = build_model(input_shape=input_shape, weights=weights, alpha=alpha, num_classes=num_classes_with_background)

    # Derive output size from the model
    model_output_shape = model.layers[-1].output.shape
    _batch, width, height, num_classes = model_output_shape

    if width != height:
        raise ValueError(f"Only square outputs are supported; not {model_output_shape}")

    output_width_height = width

    # Build weighted cross entropy loss specific to this model size
    weighted_xent = models.construct_weighted_xent_fn(model.output.shape, object_weight)
    prefetch_policy = 1 if ensure_determinism else tf.data.experimental.AUTOTUNE

    # Transform bounding box labels into segmentation maps
    def as_segmentation(ds, shuffle):
        ds = ds.map(dataset.bbox_to_segmentation(output_width_height, num_classes_with_background))
        if not ensure_determinism and shuffle:
            ds = ds.shuffle(buffer_size=batch_size * 4)
        ds = ds.batch(batch_size, drop_remainder=False).prefetch(prefetch_policy)
        return ds

    train_segmentation_dataset = as_segmentation(train_dataset, True)
    validation_segmentation_dataset = as_segmentation(validation_dataset, False)

    # Initialize bias of final classifier based on training data prior
    util.set_classifier_biases_from_dataset(model, train_segmentation_dataset)

    # Learning rate finder
    if lr_finder:
        learning_rate = ei_tensorflow.lr_finder.find_lr(model, train_segmentation_dataset, weighted_xent)

    # Compile the model
    model.compile(loss=weighted_xent, optimizer=Adam(learning_rate=learning_rate))

    # Callbacks for centroid scoring and model checkpointing
    callbacks = []
    callbacks.append(metrics.CentroidScoring(validation_dataset, output_width_height, num_classes_with_background))
    callbacks.append(metrics.PrintPercentageTrained(num_epochs))
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(best_model_path, monitor='val_f1', save_best_only=True,
                                                        mode='max', save_weights_only=True, verbose=0))

    # Train the model
    model.fit(train_segmentation_dataset, validation_data=validation_segmentation_dataset,
              epochs=num_epochs, callbacks=callbacks, verbose=0)

    # Restore the best weights
    model.load_weights(best_model_path)

    # Add explicit softmax layer before exporting the model
    softmax_layer = Softmax()(model.layers[-1].output)
    model = Model(model.input, softmax_layer)

    return model


# Training parameters
EPOCHS = args.epochs or 60
LEARNING_RATE = args.learning_rate or 0.001
BATCH_SIZE = args.batch_size or 32

# Train the model
model = train(
    num_classes=classes,
    learning_rate=LEARNING_RATE,
    num_epochs=EPOCHS,
    alpha=0.1,
    object_weight=100,
    train_dataset=train_dataset,
    validation_dataset=validation_dataset,
    best_model_path=BEST_MODEL_PATH,
    input_shape=MODEL_INPUT_SHAPE,
    batch_size=BATCH_SIZE,
    lr_finder=False,
    ensure_determinism=ensure_determinism
)
