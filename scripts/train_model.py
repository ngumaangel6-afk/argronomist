import os
import pandas as pd
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
# SCRIPT: Neural Network Training - Cassava Disease Classification
# PURPOSE: Uses Transfer Learning (MobileNetV2) to build a high-
#          accuracy classifier for agricultural leaf pathologies.
# =================================================================

# Global Hyperparameters
IMG_SIZE = (224, 224)   # Standard input resolution for MobileNetV2
BATCH_SIZE = 32         # Optimized for 8GB+ RAM
EPOCHS = 10             # Iterations over the entire dataset
LEARNING_RATE = 0.0001  # Slow rate for precise fine-tuning

def train_cassava_model(data_dir, csv_path):
    """
    Orchestrates the training pipeline: Data Loading -> Augmentation -> Compilation -> Training.
    """
    # 1. DATA ACQUISITION
    print("LOG: Fetching dataset metadata...")
    df = pd.read_csv(csv_path)
    df['label'] = df['label'].astype(str) # Keras flow_from_dataframe requires string labels
    
    # 2. DATA AUGMENTATION & PIPELINE
    # Augmentation prevents overfitting by creating variations of training images
    train_datagen = ImageDataGenerator(
        rescale=1./255,             # Rescale pixels to [0, 1]
        rotation_range=40,          # Random rotation for robustness
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2        # Reserve 20% for performance validation
    )
    
    # Training Stream
    train_generator = train_datagen.flow_from_dataframe(
        df,
        directory=data_dir,
        x_col='image_id',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    # Validation Stream
    validation_generator = train_datagen.flow_from_dataframe(
        df,
        directory=data_dir,
        x_col='image_id',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    # 3. NEURAL ARCHITECTURE (TRANSFER LEARNING)
    print("LOG: Instantiating MobileNetV2 core with ImageNet weights...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Keep pre-trained features frozen initially
    
    # Custom Prediction Layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Compress spatial features
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)             # Regularization against overfitting
    predictions = Dense(5, activation='softmax')(x) # 5 possible Cassava conditions
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compilation
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    # 4. TRAINING CALLBACKS
    # Monitor validation accuracy and save only the best iteration
    checkpoint = ModelCheckpoint('models/cassava_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
    # Stop early if training stops improving (saves time/energy)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # 5. EXECUTION
    print(f"PROCESS: Commencing training for {EPOCHS} epochs...")
    if not os.path.exists('models'): os.makedirs('models')
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop]
    )
    
    # 6. ANALYTICS VISUALIZATION
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy (Train)')
    plt.plot(history.history['val_accuracy'], label='Accuracy (Val)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss (Train)')
    plt.plot(history.history['val_loss'], label='Loss (Val)')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.savefig('models/training_curves.png')
    print("SUCCESS: Model architecture and weights archived in 'models/cassava_model.h5'")

if __name__ == "__main__":
    # Internal Path Configuration
    DATA_PATH = r"c:\Users\colli\OneDrive\Desktop\Angel\train_images"
    CSV_FILENAME = r"c:\Users\colli\OneDrive\Desktop\Angel\train.csv"
    
    if os.path.exists(DATA_PATH) and os.path.exists(CSV_FILENAME):
        train_cassava_model(DATA_PATH, CSV_FILENAME)
    else:
        print("ERROR: Dataset components missing. Verify file paths.")
