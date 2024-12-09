import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import collections
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Define the path to the dataset
dataset_path = "D:\\Programming\\Python\\cardGameProject\\dataset"

# Prepare data and labels
data = []
labels = []
label_dict = {label: idx for idx, label in enumerate(os.listdir(dataset_path))}

# Load and preprocess the images
for label in label_dict:
    label_dir = os.path.join(dataset_path, label)
    if os.path.isdir(label_dir):
        for filename in os.listdir(label_dir):
            img_path = os.path.join(label_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))  # Resize to a consistent shape
                img = img / 255.0  # Normalize the image
                data.append(img)
                labels.append(label_dict[label])

# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

class_counts = collections.Counter(labels)
print("Class Distribution:")
for label, count in class_counts.items():
    class_name = [name for name, idx in label_dict.items() if idx == label][0]  # Get the class name
    print(f"Class '{class_name}' ({label}): {count} samples")

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Calculate class weights to handle imbalanced dataset
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# Data augmentation to improve model generalization
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),  # Added contrast augmentation
    layers.RandomTranslation(0.1, 0.1)  # Added translation
])

# Build a CNN model with additional regularization
model = keras.Sequential([
    data_augmentation,  # Add data augmentation as the first layer
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),  # Add dropout after pooling
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.5),  # Increased dropout to reduce overfitting
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),  # Add dropout before the dense layer
    layers.Dense(len(label_dict), activation='softmax')  # Number of output classes
])

# Compile the model with a learning rate scheduler
initial_lr = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_lr, decay_steps=10000, decay_rate=0.9, staircase=True
)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with more epochs for better learning
history = model.fit(x_train, y_train, epochs=50, validation_split=0.1, batch_size=32, class_weight=class_weights)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc:.2f}")

# Predict on a small batch of test data
predictions = model.predict(x_test[:10])  # Adjust the slice as needed
predicted_labels = np.argmax(predictions, axis=1)

print("\nSample Predictions:")
for i in range(10):  # Adjust the range as needed
    print(f"True Label: {y_test[i]} ({[key for key, value in label_dict.items() if value == y_test[i]][0]})")
    print(f"Predicted Label: {predicted_labels[i]} ({[key for key, value in label_dict.items() if value == predicted_labels[i]][0]})\n")


# Get model predictions for the test set
y_pred = np.argmax(model.predict(x_test), axis=1)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=[key for key in label_dict]))

# Plot the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=[key for key in label_dict],
            yticklabels=[key for key in label_dict])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Save the model
model_path = "D:\\Programming\\Python\\cardGameProject\\card_classifier_model.keras"
model.save(model_path, save_format='keras')
print(f"Model saved at {model_path}")

# Visualize test images with true and predicted labels
plt.figure(figsize=(12, 6))
for i in range(10):  # Adjust the range to visualize more samples
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    plt.title(f"True: {[key for key, value in label_dict.items() if value == y_test[i]][0]}\nPred: {[key for key, value in label_dict.items() if value == predicted_labels[i]][0]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

