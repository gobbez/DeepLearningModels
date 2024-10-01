import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


def workon_iris():
    """
    This function loads the Iris array and creates a Deep Learning model to predict the name of Iris by its features
    Objective: categorize it's names
    """
    # Check if a model is already created
    check_model = 0
    select_name_model = 'iris_model.h5'
    try:
        model = load_model('iris_model.h5')
        select_new = input('Type 1 if you want to create a new model instead: ')
        if select_new == '1':
            select_name_model = input('Type the name of the new model: ')
            check_model = 0
        else:
            check_model = 1
    except:
        print('No model loaded. Proceed to creating, training and saving a new one')

    if check_model == 0:
        # Load Iris Bunch (arrays)
        iris = load_iris()

        # Convert iris to a pandas DataFrame and adds target names (species) in numers as a new column
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = iris.target

        # DataFrame analyzed, it's ready to continue

        # Set X and y
        SEED = 42
        X = df.drop('species', axis=1)
        y = df['species']
        # Set train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

        # Normalyze Data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Define Sequential model
        model = Sequential()

        # Add the first layer with 10 neurons and activation ReLU
        model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
        # Add a second layer with 8 neurons and activation ReLU
        model.add(Dense(8, activation='relu'))
        # Add a third layer as output and softmax with 3 neurons. 3 is the number of possible classifications in the dataset
        model.add(Dense(3, activation='softmax'))

        # Compile the model and use accuracy as metrics
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Define EarlyStopping to prevent to continue after it's not improving anymore (so less chance of OverFitting)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Train the model with 100 epochs
        history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping])

        # Save model
        model.save(select_name_model)


        # Plot trend of loss and accuracy for both train and test
        plt.figure(figsize=(12, 6))

        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

        # Create previsions on test data
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Check previsions with original data
        accuracy = np.mean(y_pred_classes == y_test)
        print(f'Accuracy on test set: {accuracy:.4f}')

        # Convert previsions in text
        species_names = iris.target_names
        y_test_text = [species_names[i] for i in y_test]
        y_pred_text = [species_names[i] for i in y_pred_classes]

        # Show previsisions and original data
        results = pd.DataFrame({'Actual': y_test_text, 'Predicted': y_pred_text})

        # Count total number of correct predictions
        for i in range(len(y_test_text)):
            correct = 0
            tot = 0
            if y_test_text[i] == y_pred_text[i]:
                correct += 1
            tot += 1
        correct_result = correct / tot * 100
        print(results)
        print(f"\n The final accuracy of predictions is {correct_result}")
        # 100 accuracy!!!!

workon_iris()