import os
import subprocess

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch

# ======== KONFIGURASI ========
KAGGLE_USERNAME = "" 
KAGGLE_KEY = ""
KAGGLE_DIR = os.path.expanduser("~/.kaggle")
KAGGLE_DATASET = "chetankv/dogs-cats-images"
DOWNLOAD_DIR = "data"
UNZIPPED_DIR = os.path.join(DOWNLOAD_DIR)
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
# =============================

# ======== STEP 1: Setup Kaggle Auth ========
os.makedirs(KAGGLE_DIR, exist_ok=True)
with open(os.path.join(KAGGLE_DIR, 'kaggle.json'), 'w') as f:
    f.write(f'''{{
  "username": "{KAGGLE_USERNAME}",
  "key": "{KAGGLE_KEY}"
}}''')
os.chmod(os.path.join(KAGGLE_DIR, 'kaggle.json'), 0o600)
os.environ['KAGGLE_CONFIG_DIR'] = os.path.abspath(KAGGLE_DIR)

# ======== STEP 2: Download dataset ========
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
zip_path = os.path.join(DOWNLOAD_DIR, "dogs-cats-images.zip")
if not os.path.exists(UNZIPPED_DIR):
    print("📥 Downloading dataset...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", DOWNLOAD_DIR,
        "--unzip"
    ], check=True)
else:
    print("✅ Dataset already downloaded.")

# ======== STEP 3: Image Generator Setup ========
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_generator = datagen.flow_from_directory(
    os.path.join(UNZIPPED_DIR, "dataset", "training_set"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

val_generator = datagen.flow_from_directory(
    os.path.join(UNZIPPED_DIR, "dataset", "test_set"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
)

# ======== STEP 4: Build Model Function (with HPT, Dropout, BN, Transfer Learning) ========
def build_model(hp):
    base_model = MobileNetV2(include_top=False, input_shape=(160, 160, 3), weights="imagenet")
    base_model.trainable = False

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1)))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=optimizers.Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model

# ======== STEP 5: Hyperparameter Tuning ========
tuner = RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=5,
    executions_per_trial=1,
    directory="tuner_logs",
    project_name="cat_dog_classifier",
)

early_stop = EarlyStopping(monitor="val_loss", patience=3)

tuner.search(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[early_stop],
)

# ======== STEP 6: Final Training with Best Model ========
best_model = tuner.get_best_models(num_models=1)[0]
history = best_model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[early_stop],
)

best_model.save("cat_dog_classifier.h5")
print("✅ Model saved as cat_dog_classifier.h5")
