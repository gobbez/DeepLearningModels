import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from keras import models
from keras import layers
from keras import callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
import os

from plots.show_plots import ShowPlot
from plots.show_network import ShowModel


def load_dataset():
    """Load dataset, set and normalize variables"""
    # Load dataset
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Normalize data
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    # Convert labels in one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return X_train, X_test, y_train, y_test


def check_model():
    """Check if a model is already created, else creates a Deep Learning model"""
    try:
        if os.path.exists(model_name):
            # Skip to Previsions
            model = models.load_model(model_name)
            print('Model loaded.. proceed with prediction')
            return model
        else:
            print('No model found, proceed with Deep Learning')
            deeplearning()
            model = models.load_model(model_name)
            print('Model loaded.. proceed with prediction')
            return model
    except:
        print('Failed to load model, proceed with Deep Learning')
        model = deeplearning()


def deeplearning():
    """
    Use Deep Learning to categorize images
    """
    # Define model Sequential CNN
    model = models.Sequential()

    # Add levels
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))
    # Level fully connected
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))

    # Show model
    ShowModel(model)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define EarlyStopping to prevent overfitting
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train model
    history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test),
                        callbacks=[early_stopping])

    # SAVE MODEL
    model.save(model_name)
    print('Model Saved... proceed with showing plots')

    # Create plots
    plot = ShowPlot()
    plot.lineplot(history.history['loss'], history.history['val_loss'], 'Loss', 'Epochs', 'Loss', True)
    plot.lineplot(history.history['accuracy'], history.history['val_accuracy'], 'Accuracy', 'Epochs',
                  'Accuracy', True)
    print('Proceeding loading the model for previsions..')


def previsions():
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc:.4f}')

    # Show previsions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)

    # Show images with previsions
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_test[i])
        plt.xlabel(f'Pred: {predicted_classes[i]} | True: {true_classes[i]}')
    plt.savefig(fname="img_predict.png")
    plt.show()

    select_image = input('Type 1 if you want to proceed predicting your image: ')
    if select_image == '1':
        # Continue to predict a new image (loaded by user)
        # Load the new image
        img_path = 'insert_img_path'
        img = image.load_img(img_path, color_mode='grayscale', target_size=(28, 28))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the image

        # Predict
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        list_classifications = ['T - shirt / top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        # Show predicted class
        print(f'Predicted class: {list_classifications[predicted_class[0]]}')



# Start
model_name = 'fashion_mnist_model.keras'

# Load values
X_train, X_test, y_train, y_test = load_dataset()
# Check if model is already present, else create a new Deep Learning model
model = check_model()
# Check previsions and %
previsions()