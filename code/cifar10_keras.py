import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image


def workon_cifar10():
    """
    Use Deep Learning to categorize images
    """
    # Check if a model is already created
    check_model = 0
    select_name_model = 'cifar10_model.h5'
    try:
        model = load_model('cifar10_model.h5')
        select_new = input('Type 1 if you want to create a new model instead: ')
        if select_new == '1':
            select_name_model = input('Type the name of the new model: ')
            check_model = 0
        else:
            check_model = 1
    except:
        print('No model loaded. Proceed to creating, training and saving a new one')

    if check_model == 0:
        # Load dataset CIFAR-10
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        # Normalize data
        X_train = X_train.astype('float32') / 255
        X_test = X_test.astype('float32') / 255

        # Convert labels in one-hot encoding
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

        # Define model Sequential CNN
        model = Sequential()

        # Add levels
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # Level fully connected
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Define EarlyStopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train model
        history = model.fit(X_train, y_train, batch_size=64, epochs=100, validation_data=(X_test, y_test),
                            callbacks=[early_stopping])

        # SAVE MODEL
        model.save(select_name_model)


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
        plt.show()

    select_image = input('Type 1 if you want to proceed predicting your image: ')
    if select_image == '1':
        # Continue to predict a new image (loaded by user)
        # Load the new image
        img_path = 'insert_img_path'
        img = image.load_img(img_path, target_size=(32, 32))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize the image

        # Predict
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)
        list_classifications = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # Show predicted class
        print(f'Predicted class: {list_classifications[predicted_class[0]]}')


workon_cifar10()