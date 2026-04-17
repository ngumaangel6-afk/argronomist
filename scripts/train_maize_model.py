import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# =================================================================
# SCRIPT: Maize Disease Training Pipeline (Folder-Based)
# PURPOSE: Trains a classifier using folder-organized datasets.
# =================================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

def train_maize_model(data_dir):
    """
    Directly consumes images from subdirectories representing classes.
    """
    # 1. FLOW FROM DIRECTORY
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.15,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # 2. MODEL CONFIGURATION
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False 
    
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(train_gen.num_classes, activation='softmax')(x) 
    
    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    # 3. DIRECTORY SETUP
    if not os.path.exists('models'): os.makedirs('models')

    # 4. TRAINING
    print(f"PROCESS: Training on {train_gen.num_classes} Maize conditions...")
    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=[
            ModelCheckpoint('models/maize_model.h5', save_best_only=True),
            EarlyStopping(patience=3)
        ]
    )
    print("SUCCESS: Maize model archived.")

if __name__ == "__main__":
    MAIZE_ROOT = r"c:\Users\colli\OneDrive\Desktop\Angel\Multicrop-Disease-Maiz Disease-Pests and disease"
    if os.path.exists(MAIZE_ROOT):
        train_maize_model(MAIZE_ROOT)
    else:
        print("ERROR: Maize dataset folder not found.")
