import os
import subprocess
import shutil
import random

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
UNKNOWN_DATASET = "ezzzio/random-images"
DOWNLOAD_DIR = "data"
UNZIPPED_DIR = os.path.join(DOWNLOAD_DIR, "dataset")
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
# =============================
# a. Hyperparameter tuning: menggunakan RandomSearch dari Keras Tuner
# b. Dropout: ditambahkan di arsitektur model, dituning nilai dropout-nya
# c. Batch normalization: ditambahkan setelah GlobalAveragePooling2D
# d. Transfer learning: menggunakan MobileNetV2 (pre-trained ImageNet)
# =============================
# ======== STEP 1: Setup Kaggle Auth ========
# Menulis file konfigurasi autentikasi Kaggle agar dapat digunakan oleh API
os.makedirs(KAGGLE_DIR, exist_ok=True)
with open(os.path.join(KAGGLE_DIR, 'kaggle.json'), 'w') as f:
    f.write(f'''{{
  "username": "{KAGGLE_USERNAME}",
  "key": "{KAGGLE_KEY}"
}}''')
os.chmod(os.path.join(KAGGLE_DIR, 'kaggle.json'), 0o600)
os.environ['KAGGLE_CONFIG_DIR'] = os.path.abspath(KAGGLE_DIR)

# ======== STEP 2: Download dataset ========
# Mengunduh dan mengekstrak dataset utama dari Kaggle
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
if not os.path.exists(os.path.join(UNZIPPED_DIR, "__done_main_dataset")):
    print("📥 Downloading dataset...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", DOWNLOAD_DIR,
        "--unzip"
    ], check=True)
    # Pindahkan struktur folder
    if os.path.exists(os.path.join(DOWNLOAD_DIR, "dogs-cats-images", "training_set")):
        shutil.move(os.path.join(DOWNLOAD_DIR, "dogs-cats-images", "training_set"), UNZIPPED_DIR)
        shutil.move(os.path.join(DOWNLOAD_DIR, "dogs-cats-images", "test_set"), UNZIPPED_DIR)
        shutil.rmtree(os.path.join(DOWNLOAD_DIR, "dogs-cats-images"))
    with open(os.path.join(UNZIPPED_DIR, "__done_main_dataset"), 'w') as f:
        f.write("done")
else:
    print("✅ Dataset already downloaded.")

# ======== STEP 3: Download Unknown Class Images ========
# Mengunduh dataset kelas unknown secara terpisah
# Kelas ini untuk dataset selain cat dan dog
UNKNOWN_DIR = DOWNLOAD_DIR
if not os.path.exists(os.path.join(UNKNOWN_DIR, "__done_unknown_download")):
    print("Downloading unknown class images...")
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", UNKNOWN_DATASET,
        "-p", UNKNOWN_DIR,
        "--unzip"
    ], check=True)
    with open(os.path.join(UNKNOWN_DIR, "__done_unknown_download"), 'w') as f:
        f.write("done")
else:
    print("Unknown dataset already downloaded.")

# ======== STEP 3.1: Temukan folder gambar unknown secara otomatis ========
# Deteksi otomatis folder yang berisi gambar dari dataset unknown
def find_image_folder(base_path):
    for root, dirs, files in os.walk(base_path):
        image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if image_files:
            print(f"Ditemukan folder berisi gambar: {root} ({len(image_files)} file)")
            return root
    raise ValueError("Tidak ditemukan gambar di dataset unknown.")

ACTUAL_UNKNOWN_IMAGES = find_image_folder(UNKNOWN_DIR)

# ======== STEP 4: Move Unknown Images ========
# Membagi gambar unknown ke training dan testing secara acak
def move_unknown_images(src_dir, train_dst, test_dst, train_ratio=0.8):
    os.makedirs(train_dst, exist_ok=True)
    os.makedirs(test_dst, exist_ok=True)
    all_images = [f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_ratio)
    train_imgs = all_images[:split_idx]
    test_imgs = all_images[split_idx:]
    for img in train_imgs:
        shutil.copy(os.path.join(src_dir, img), os.path.join(train_dst, img))
    for img in test_imgs:
        shutil.copy(os.path.join(src_dir, img), os.path.join(test_dst, img))
    print(f"Copied {len(train_imgs)} to {train_dst}")
    print(f"Copied {len(test_imgs)} to {test_dst}")

move_unknown_images(
    src_dir=ACTUAL_UNKNOWN_IMAGES,
    train_dst=os.path.join(UNZIPPED_DIR, "training_set", "unknown"),
    test_dst=os.path.join(UNZIPPED_DIR, "test_set", "unknown")
)

# ======== STEP 5: Image Generator Setup ========
# Preprocessing dan augmentasi gambar untuk pelatihan
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True,
)

train_generator = datagen.flow_from_directory(
    os.path.join(UNZIPPED_DIR, "training_set"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

val_generator = datagen.flow_from_directory(
    os.path.join(UNZIPPED_DIR, "test_set"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

# ======== STEP 6: Build Model Function ========
# Fungsi pembuat model dengan komponen:
# - MobileNetV2 pretrained (transfer learning)
# - GlobalAveragePooling2D untuk mereduksi dimensi
# - BatchNormalization untuk stabilisasi pelatihan
# - Dropout sebagai regularisasi (nilai dituning)
# - Dense(3) untuk klasifikasi 3 kelas (cat, dog, unknown)
def build_model(hp):
    base_model = MobileNetV2(include_top=False, input_shape=(160, 160, 3), weights="imagenet")
    base_model.trainable = False

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(hp.Float("dropout", 0.2, 0.5, step=0.1)))
    model.add(layers.Dense(3, activation="softmax"))

    model.compile(
        optimizer=optimizers.Adam(hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

# ======== STEP 7: Hyperparameter Tuning ========
# Menjalankan RandomSearch untuk tuning dropout dan learning rate
tuner = RandomSearch(
    build_model,
    objective="val_accuracy",
    max_trials=5,
    executions_per_trial=1,
    directory="tuner_logs",
    project_name="cat_dog_unknown_classifier",
)

early_stop = EarlyStopping(monitor="val_loss", patience=3)

tuner.search(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[early_stop],
)

# ======== STEP 8: Final Training with Best Model ========
# Melatih ulang model terbaik dan menyimpan hasilnya
best_model = tuner.get_best_models(num_models=1)[0]
history = best_model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[early_stop],
)

best_model.save("cat_dog_unknown_classifier.h5")
print("Model saved as cat_dog_unknown_classifier.h5")