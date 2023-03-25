def import_tensorflow():
    # Filter tensorflow version warnings
    import os
    # https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
    import warnings
    # https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=Warning)
    import tensorflow as tf
    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(0)
    import logging
    tf.get_logger().setLevel(logging.ERROR)
    return tf
tf = import_tensorflow()

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

from tensorflow import keras
import segmentation_models as sm
from keras import layers
from keras.models import Sequential
import segmentation_models as sm
import keras
from tqdm import tqdm


def train_models(epochs, backbones, x_train, y_train, x_val, y_val, to_save, architecture, learning_rate=0.01, callbacks=None, batch_size=16):
    for backbone in tqdm(backbones):
        BACKBONE = backbone
        to_save_name = "model_linknet" + BACKBONE + "_10_epochs"
        preprocess_input = sm.get_preprocessing(BACKBONE)
        
        data_augmentation = Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2)])

        x_train = data_augmentation(x_train)
        y_train = data_augmentation(y_train)


        x_train = preprocess_input(x_train)
        x_val = preprocess_input(x_val)

        # define model
        if architecture == "Unet":
            model = sm.Unet(BACKBONE, encoder_weights="imagenet")
        elif architecture == "Linknet":
            model = sm.Linknet(BACKBONE, encoder_weights="imagenet")
        elif architecture == "FPN":
            model = sm.FPN(BACKBONE, encoder_weights="imagenet")
        elif architecture == "PSPNet":
            model = sm.PSPNet(BACKBONE, encoder_weights="imagenet")    
            
        model.compile(
            keras.optimizers.Adam(learning_rate=learning_rate),
            loss=sm.losses.bce_jaccard_loss,
            metrics=["accuracy", sm.metrics.iou_score],
        )

        if callbacks is not None:
            callbacks = callbacks

        history = model.fit(x_train, 
                        y_train, 
                        batch_size=batch_size, 
                        epochs=epochs,
                        validation_data=(x_val, y_val), 
                        callbacks=callbacks)
        
        to_save[backbone] = history