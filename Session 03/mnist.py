import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load MNIST (with ARFF parser to avoid pandas error)
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
X, y = mnist["data"], mnist["target"].astype(np.int8)

print("Dataset shape:", X.shape, y.shape)

# SHOW 1: First 5 raw images
fig, axes = plt.subplots(1, 5, figsize=(12, 2))
print("\n=== RAW IMAGES (0-255 pixels) ===")
for i in range(5):
    img = X[i].reshape(28, 28)
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'Label: {y[i]}')
    axes[i].axis('off')
plt.suptitle('First 5 MNIST Images (Raw)')
plt.tight_layout()
plt.show()

# 2. Train-test split + scaling
X_train, X_test = X[:60000]/255.0, X[60000:]/255.0
y_train, y_test = y[:60000], y[60000:]

print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

# SHOW 2: First 5 scaled training images
fig, axes = plt.subplots(1, 5, figsize=(12, 2))
print("\n=== SCALED IMAGES (0-1 range) ===")
for i in range(5):
    img = X_train[i].reshape(28, 28)
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'Label: {y_train[i]}\nMin: {img.min():.2f}, Max: {img.max():.2f}')
    axes[i].axis('off')
plt.suptitle('First 5 Training Images (Scaled)')
plt.tight_layout()
plt.show()

# 3. FIXED Logistic Regression (no multi_class parameter)
print("\nTraining Logistic Regression...")
clf = LogisticRegression(
    solver="lbfgs",      # Handles multiclass automatically now
    max_iter=1000,
    n_jobs=-1            # Parallel processing
)
clf.fit(X_train, y_train)
print("Training complete!")

# 4. Predictions
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

# SHOW 3: First 5 test predictions
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
print("\n=== TEST PREDICTIONS ===")
for i in range(5):
    img = X_test[i].reshape(28, 28)
    color = 'green' if y_pred[i] == y_test[i] else 'red'
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'True: {y_test[i]}\nPred: {y_pred[i]}\n{"✓" if y_pred[i]==y_test[i] else "✗"}', 
                     color=color, fontsize=10)
    axes[i].axis('off')
plt.suptitle(f'First 5 Test Predictions (Accuracy: {acc:.1%})')
plt.tight_layout()
plt.show()
