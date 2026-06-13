# ============================================
# Dogs vs Cats CNN Classifier
# Slash Mark Internship - Task 2
# Name: Rohan Gunjal
# ============================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)

# ---- STEP 1: Download Dataset ----
print("\n📥 Downloading Dogs vs Cats dataset...")
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# We'll use CIFAR-10 which has dogs and cats
# Class 3 = cat, Class 5 = dog
cat_indices = np.where(train_labels == 3)[0][:5000]
dog_indices = np.where(train_labels == 5)[0][:5000]

train_images_filtered = np.vstack([
    train_images[cat_indices],
    train_images[dog_indices]
])
train_labels_filtered = np.hstack([
    np.zeros(len(cat_indices)),
    np.ones(len(dog_indices))
])

# Shuffle
shuffle_idx = np.random.permutation(len(train_labels_filtered))
train_images_filtered = train_images_filtered[shuffle_idx]
train_labels_filtered = train_labels_filtered[shuffle_idx]

# Test set
test_cat_indices = np.where(test_labels == 3)[0][:1000]
test_dog_indices = np.where(test_labels == 5)[0][:1000]

test_images_filtered = np.vstack([
    test_images[test_cat_indices],
    test_images[test_dog_indices]
])
test_labels_filtered = np.hstack([
    np.zeros(len(test_cat_indices)),
    np.ones(len(test_dog_indices))
])

print(f"✅ Training images: {train_images_filtered.shape}")
print(f"✅ Test images: {test_images_filtered.shape}")

# ---- STEP 2: Normalize Images ----
print("\n🔧 Preprocessing images...")
train_images_normalized = train_images_filtered / 255.0
test_images_normalized = test_images_filtered / 255.0

# ---- STEP 3: Build CNN Model ----
print("\n🏗️ Building CNN architecture...")
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# ---- STEP 4: Compile Model ----
print("📊 Compiling model...")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print model summary
print("\n" + "="*50)
model.summary()
print("="*50)

# ---- STEP 5: Train Model ----
print("\n🚀 Training model (this may take 2-3 minutes)...")
history = model.fit(
    train_images_normalized, train_labels_filtered,
    epochs=15,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ---- STEP 6: Evaluate Model ----
print("\n📈 Evaluating on test set...")
test_loss, test_accuracy = model.evaluate(test_images_normalized, test_labels_filtered)
print(f"\n✅ Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

# ---- STEP 7: Plot Training History ----
print("\n📊 Plotting training history...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('training_history.png')
print("✅ Graph saved as 'training_history.png'")
plt.show()

# ---- STEP 8: Test Predictions ----
print("\n🎯 Testing predictions on sample images...")
predictions = model.predict(test_images_normalized[:10])

for i in range(10):
    if predictions[i][0] > 0.5:
        print(f"Image {i}: DOG (confidence: {predictions[i][0]*100:.2f}%)")
    else:
        print(f"Image {i}: CAT (confidence: {(1-predictions[i][0])*100:.2f}%)")

print("\n" + "="*50)
print("✅ TASK 2 COMPLETE!")
print("="*50)