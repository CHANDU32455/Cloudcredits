# Handwritten Digit Recognition using CNN
# Dataset: MNIST
# Evaluation: Accuracy, Confusion Matrix, Classification Report

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import random
import os

# Create output folder
output_dir = 'mnist_digit_classifier'
os.makedirs(output_dir, exist_ok=True)

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train_one_hot = to_categorical(y_train, num_classes=10)
y_test_one_hot = to_categorical(y_test, num_classes=10)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Display model architecture
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Train the model
history = model.fit(
    x_train, y_train_one_hot,
    epochs=5, batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop]
)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test_one_hot, verbose=0)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Predict and analyze
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig(os.path.join(output_dir, 'training_history.png'))
plt.show()

# Visualize random predictions
idx = random.sample(range(len(x_test)), 10)
plt.figure(figsize=(12, 4))
for i, index in enumerate(idx):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
    plt.title(f'True: {y_test[index]}\nPred: {y_pred_classes[index]}')
    plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'sample_predictions.png'))
plt.show()

# Save model in H5 format
model.save(os.path.join(output_dir, 'mnist_cnn_model.h5'))
print("Model saved as 'mnist_cnn_model.h5'")

# Export to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(os.path.join(output_dir, 'mnist_model.tflite'), 'wb') as f:
    f.write(tflite_model)
print("Model exported as 'mnist_model.tflite'")

# Simple inference test
print("\nTesting a sample image:")
sample = x_test[0].reshape(1, 28, 28, 1)
prediction = model.predict(sample)
print("Predicted digit:", np.argmax(prediction))
plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.title(f"Model Prediction: {np.argmax(prediction)}")
plt.axis('off')
plt.show()

