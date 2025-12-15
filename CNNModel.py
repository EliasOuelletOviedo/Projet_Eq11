import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.layers as L

from ProgressBarCallback import TQDMProgressBar

def RunCNN(X_train, X_test, y_train, y_test, cnn_model_path):
    print("Training CNN model...")

    cnn_model = tf.keras.Sequential([
        L.Conv1D(128, kernel_size=7, strides=1, padding='same', activation='relu', input_shape=(X_train.shape[1],1)),
        L.BatchNormalization(),
        L.MaxPool1D(pool_size=4, strides=2, padding='same'),

        L.Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'),
        L.BatchNormalization(),
        L.MaxPool1D(pool_size=4, strides=2, padding='same'),
        L.Dropout(0.2),

        L.Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
        L.BatchNormalization(),
        L.MaxPool1D(pool_size=3, strides=2, padding='same'),
        L.Dropout(0.2),

        L.Flatten(),
        L.Dense(256, activation='relu'),
        L.BatchNormalization(),
        L.Dropout(0.3),

        L.Dense(22, activation='softmax')
    ])

    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    cnn_checkpoint = ModelCheckpoint(cnn_model_path, monitor='val_accuracy', save_best_only=True, verbose=0)
    early_stop = EarlyStopping(monitor='val_loss', patience=7, verbose=0)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0)

    progress_callback = TQDMProgressBar()

    cnn_history = cnn_model.fit(
        X_train, y_train,
        epochs=64,
        validation_data=(X_test, y_test),
        batch_size=64,
        callbacks=[early_stop, lr_reduction, cnn_checkpoint, progress_callback],
        shuffle=True,
        verbose=0
    )

    print("CNN model training completed.")
    print(f"CNN model saved to {cnn_model_path}.")

    epochs = range(len(cnn_history.history['accuracy']))

    fig, ax = plt.subplots(1, 2, figsize=(20,6))

    ax[0].plot(epochs, cnn_history.history['loss'], label='Training Loss')
    ax[0].plot(epochs, cnn_history.history['val_loss'], label='Validation Loss')
    ax[0].set_title('Training & Validation Loss')
    ax[0].legend()

    ax[1].plot(epochs, cnn_history.history['accuracy'], label='Training Accuracy')
    ax[1].plot(epochs, cnn_history.history['val_accuracy'], label='Validation Accuracy')
    ax[1].set_title('Training & Validation Accuracy')
    ax[1].legend()

    plt.show()