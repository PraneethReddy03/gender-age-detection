import os
import tensorflow as tf
from tensorflow.keras import layers, models
import argparse

def build_model(input_shape, n_classes):
    inp = layers.Input(input_shape)
    x = layers.Conv2D(32, 3, activation='relu')(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)

    # Gender head
    g = layers.Dense(64, activation='relu')(x)
    g_out = layers.Dense(1, activation='sigmoid', name="gender")(g)

    # Age head
    a = layers.Dense(64, activation='relu')(x)
    a_out = layers.Dense(n_classes, activation='softmax', name="age")(a)

    model = models.Model(inputs=inp, outputs=[g_out, a_out])
    model.compile(
        optimizer='adam',
        loss={"gender": "binary_crossentropy", "age": "categorical_crossentropy"},
        metrics={"gender": "accuracy", "age": "accuracy"}
    )
    return model

def main(data_dir, epochs, batch_size):
    # TODO: load TF Dataset or generators from data_dir
    input_shape = (64, 64, 3)
    n_age_classes = 8
    model = build_model(input_shape, n_age_classes)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "outputs/models/best.h5", save_best_only=True
    )
    # TODO: replace train_ds/val_ds with real datasets
    # model.fit(train_ds, validation_data=val_ds,
    #           epochs=epochs, callbacks=[checkpoint], batch_size=batch_size)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="Processed images folder")
    p.add_argument("--epochs",    type=int, default=20)
    p.add_argument("--batch_size",type=int, default=32)
    args = p.parse_args()
    main(args.data_dir, args.epochs, args.batch_size)
ø
ø

