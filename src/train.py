import os
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

def build_model(input_shape, n_age_classes):
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
    a_out = layers.Dense(n_age_classes, activation='softmax', name="age")(a)

    model = models.Model(inputs=inp, outputs=[g_out, a_out])
    model.compile(
        optimizer='adam',
        loss={"gender": "binary_crossentropy", "age": "sparse_categorical_crossentropy"},
        metrics={"gender": "accuracy", "age": "accuracy"}
    )
    return model

def parse_filename(fp):
    # Extract just the filename (after the last slash)
    filename = tf.strings.split(fp, os.sep)[-1]
    # Now split on underscores: '23_0_...' → ['23','0',...]
    parts = tf.strings.split(filename, '_')
    age = tf.strings.to_number(parts[0], out_type=tf.int32)
    gender = tf.strings.to_number(parts[1], out_type=tf.int32)
    age_bracket = age // 10
    return age, gender, age_bracket

def load_and_preprocess(fp):
    img = tf.io.read_file(fp)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [64, 64]) / 255.0
    age, gender, age_bracket = parse_filename(fp)
    return img, {"gender": gender, "age": age_bracket}

def get_datasets(data_dir, batch_size, split=0.8):
    files = tf.data.Dataset.list_files(os.path.join(data_dir, '*.jpg'))
    ds = files.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000, reshuffle_each_iteration=False)
    total = len(tf.io.gfile.glob(os.path.join(data_dir, '*.jpg')))
    train_count = int(split * total)
    train_ds = ds.take(train_count).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds   = ds.skip(train_count).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

def main(data_dir, epochs, batch_size):
    print(f"DEBUG: Running training on '{data_dir}' for {epochs} epochs, batch size {batch_size}")
    train_ds, val_ds = get_datasets(data_dir, batch_size)
    total = len(tf.io.gfile.glob(os.path.join(data_dir, '*.jpg')))
    train_count = int(0.8 * total)
    val_count   = total - train_count
    print(f"DEBUG: {total} total images → {train_count} train / {val_count} val")

    model = build_model((64, 64, 3), n_age_classes=10)
    ckpt = callbacks.ModelCheckpoint(
        "outputs/models/best.h5",
        save_best_only=True,
        monitor="val_gender_accuracy",
        mode="max"
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[ckpt]
    )

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   required=True, help="Folder with processed images")
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()
    main(args.data_dir, args.epochs, args.batch_size)

