# deep learnig models for trading and mid-price.

from tensorflow import keras
import tensorflow as tf


# model for trading
def cnn_classification_trading_model(look_back, data_level, learning_rate):
    input_price = keras.Input(shape=(look_back, int(2*data_level)), name='price')
    input_volume = keras.Input(shape=(look_back, int(2*data_level)), name='volume')

    x_price = keras.backend.expand_dims(input_price)
    x_volume = keras.backend.expand_dims(input_volume)
    x_mix = keras.layers.Multiply()([x_price, x_volume])

    x_price = keras.layers.Conv2D(16, (7, 4), activation='relu')(x_price)
    x_price = keras.layers.MaxPooling2D((3, 3))(x_price)
    x_price = keras.layers.Flatten()(x_price)
    x_price = keras.layers.Dropout(0.2)(x_price)
    x_price = keras.layers.Dense(8)(x_price)

    x_volume = keras.layers.Conv2D(16, (7, 4), activation='relu')(x_volume)
    x_volume = keras.layers.MaxPooling2D((3, 3))(x_volume)
    x_volume = keras.layers.Flatten()(x_volume)
    x_volume = keras.layers.Dropout(0.2)(x_volume)
    x_volume = keras.layers.Dense(8)(x_volume)

    x_mix = keras.layers.Conv2D(16, (7, 4), activation='relu')(x_mix)
    x_mix = keras.layers.MaxPooling2D((3, 3))(x_mix)
    x_mix = keras.layers.Flatten()(x_mix)
    x_mix = keras.layers.Dropout(0.2)(x_mix)
    x_mix = keras.layers.Dense(8)(x_mix)

    x = keras.layers.Concatenate()([x_price, x_volume, x_mix])
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs=[input_price, input_volume], outputs=outputs)
    adam_optimizer = keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
    model.compile(optimizer=adam_optimizer, loss='binary_crossentropy', metrics=[
                  keras.metrics.AUC(), 'accuracy'])

    return model



    