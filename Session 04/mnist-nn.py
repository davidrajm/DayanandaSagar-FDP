import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow/Keras version:", tf.__version__)

# 1. Load SAME MNIST data (already working!)
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
X, y = mnist["data"], mnist["target"].astype(np.int8)

# Reshape to proper image shape: (N, 28, 28, 1)
X = X.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y = keras.utils.to_categorical(y, 10)  # One-hot encode labels

# Train-test split
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

print("X_train shape:", X_train.shape)  # (60000, 28, 28, 1)
print("y_train shape:", y_train.shape)  # (60000, 10)

# SHOW 1: First 5 images (same as before)
fig, axes = plt.subplots(1, 5, figsize=(12, 2))
print("\n=== DEEP LEARNING INPUT IMAGES ===")
for i in range(5):
    axes[i].imshow(X_train[i, :, :, 0], cmap='gray')
    axes[i].set_title(f'Label: {np.argmax(y_train[i])}')
    axes[i].axis('off')
plt.suptitle('First 5 Images for Neural Network')
plt.tight_layout()
plt.show()

# 2. Build SIMPLE Deep Learning Model
print("\n=== BUILDING NEURAL NETWORK ===")
model = keras.Sequential([
    # Input: 28x28x1 images
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),  # Flatten to 1D
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Prevent overfitting
    
    layers.Dense(10, activation='softmax')  # 10 classes
])

model.summary()

# 3. Compile & Train
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nTraining Deep Learning Model...")
history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=5,  
    validation_split=0.1,
    verbose=1
)

# 4. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nDeep Learning Test Accuracy: {test_acc:.4f}")

# SHOW 2: Training history plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()

# 5. Predictions & Visualization
y_pred_proba = model.predict(X_test[:5])
y_pred = np.argmax(y_pred_proba, axis=1)

# SHOW 3: First 5 predictions
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
print("\n=== DEEP LEARNING PREDICTIONS ===")
for i in range(5):
    img = X_test[i, :, :, 0]
    true_label = np.argmax(y_test[i])
    pred_label = y_pred[i]
    confidence = y_pred_proba[i, pred_label]
    
    color = 'green' if pred_label == true_label else 'red'
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1%}', 
                     color=color, fontsize=10)
    axes[i].axis('off')
plt.suptitle(f'Deep Learning Predictions (Test Acc: {test_acc:.1%})')
plt.tight_layout()
plt.show()
