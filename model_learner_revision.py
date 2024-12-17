import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Path ke dataset
dataset_path = "D:\\Programming\\Python\\cardGameProject\\kaggle_set\\train"  # Pastikan path ini sesuai dengan lokasi dataset Anda
os.chdir("D:\\Programming\\Python\\cardGameProject\\kaggle_set\\train")

# Data dan label
data = []
labels = []

# Buat label dictionary berdasarkan subdirektori (kombinasi jenis dan nilai kartu)
label_dict = {label: idx for idx, label in enumerate(os.listdir(dataset_path))}

# Load data dan buat label berdasarkan subdirektori
for label in label_dict:
    label_dir = os.path.join(dataset_path, label)
    if os.path.isdir(label_dir):
        for filename in os.listdir(label_dir):
            img_path = os.path.join(label_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (190, 338))  # Ubah ukuran gambar ke dimensi tetap
                img = img / 255.0  # Normalisasi ke rentang [0, 1]
                data.append(img)
                labels.append(label_dict[label])


# Convert data dan labels ke numpy arrays
data = np.array(data, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)

# Split data menjadi train dan test
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Menghitung class weights
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# Data augmentation untuk meningkatkan generalisasi
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)
])
# Model CNN
model = keras.Sequential([
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(190, 338, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(label_dict), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train, epochs=30, validation_split=0.1, batch_size=8, class_weight=class_weights)

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.2f}")

# Predict
y_pred = np.argmax(model.predict(x_test), axis=1)

# Classification report dan confusion matrix dengan semua label
all_labels = np.arange(len(label_dict))  # memastikan semua kelas terdaftar dalam laporan dan matriks
print("Classification Report:")
print(classification_report(y_test, y_pred, labels=all_labels, target_names=[key for key in label_dict], zero_division=1))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=all_labels)
plt.figure(figsize=(15, 12))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[key for key in label_dict],
            yticklabels=[key for key in label_dict])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save model
model_path = "new18_card_classifier_model.keras"
model.save(model_path)
print(f"Model saved at {model_path}")