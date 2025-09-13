# Waste-Segregation-Identifier
# ===============================================
# 1. Import Libraries
# ===============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ===============================================
# 2. Dataset Setup
# ===============================================
train_dir = "/content/dataset/train"   # change if different
test_dir  = "/content/dataset/test"

# Image preprocessing & augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

test_set = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical'
)

print("Classes:", train_set.class_indices)

# ===============================================
# 3. CNN Model Implementation
# ===============================================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_set.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ===============================================
# 4. Model Training
# ===============================================
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_waste_cnn_model.h5", save_best_only=True, monitor="val_accuracy")

history = model.fit(
    train_set,
    validation_data=val_set,
    epochs=20,
    callbacks=[early_stop, checkpoint]
)

# ===============================================
# 5. Performance Visualization
# ===============================================
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

# ===============================================
# 6. Hyperparameter Tuning (Simple Example)
# ===============================================
# Adjusting learning rate & batch size manually (example)
from tensorflow.keras.optimizers import Adam

model_tuned = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_set.num_classes, activation='softmax')
])

model_tuned.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

history_tuned = model_tuned.fit(
    train_set,
    validation_data=val_set,
    epochs=15,
    callbacks=[early_stop]
)

# ===============================================
# 7. Save & Load Model
# ===============================================
model_tuned.save("waste_cnn_model_final.h5")
print("✅ Final model saved as waste_cnn_model_final.h5")

loaded_model = load_model("waste_cnn_model_final.h5")

# ===============================================
# 8. Prediction on New Image
# ===============================================
from tensorflow.keras.preprocessing import image

img_path = "/content/sample_image.jpg"  # provide path of a test image
img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = loaded_model.predict(img_array)
predicted_class = np.argmax(prediction)
class_labels = list(train_set.class_indices.keys())

print("🔮 Predicted Class:", class_labels[predicted_class])

# ===============================================
# 9. Conclusion
# ===============================================
print("""
✅ CNN model trained successfully for Waste Classification.
✅ Final accuracy: {:.2f}% on validation set.
✅ Model saved and can be deployed.
Hurrah! Capstone project completed successfully 🚀
""".format(max(history.history['val_accuracy'])*100))
