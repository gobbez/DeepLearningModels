import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import models
from keras import layers
from keras import callbacks
import os

from plots.show_plots import ShowPlot
from plots.show_network import ShowModel


def load_dataset():
    """Load Iris dataset, set and normalize variables"""
    # Load Iris Dataset
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

    # Normalize Data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return iris, SEED, X, y, X_train, X_test, y_train, y_test


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
    This function loads the Iris array and creates a Deep Learning model to predict the name of Iris by its features.
    After the Deep Learning model, there will be displayed plots and reloaded the model for previsions
    Objective: categorize Iris names
    """
    # Define Sequential model
    model = models.Sequential()

    # Add the first layer with 10 neurons and activation ReLU
    model.add(layers.Dense(10, input_dim=X_train.shape[1], activation='relu'))
    # Add a second layer with 8 neurons and activation ReLU
    model.add(layers.Dense(8, activation='relu'))
    # Add a third layer as output and softmax with 3 neurons. 3 is the number of possible classifications in the dataset
    model.add(layers.Dense(3, activation='softmax'))

    # Show model
    ShowModel(model)

    # Compile the model and use accuracy as metrics
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define EarlyStopping to prevent to continue after it's not improving anymore (so less chance of OverFitting)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model with 100 epochs
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Save model
    model.save(model_name)
    print('Model Saved... proceed with showing plots')

    # Create plots
    plot = ShowPlot()
    plot.lineplot(history.history['loss'], history.history['val_loss'], 'Loss', 'Epochs', 'Loss', True)
    plot.lineplot(history.history['accuracy'], history.history['val_accuracy'], 'Accuracy', 'Epochs',
                        'Accuracy', True)
    print('Proceeding loading the model for previsions..')


def previsions():
    """Create previsions on test data"""
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
    correct = sum(1 for i in range(len(y_test_text)) if y_test_text[i] == y_pred_text[i])
    percentage = round(correct / len(y_test_text) * 100, 1)
    print(results)
    print(f"\n The final accuracy of predictions is {percentage}%")


# Start
model_name = 'iris_model.keras'

# Load values
iris, SEED, X, y, X_train, X_test, y_train, y_test = load_dataset()
# Check if model is already present, else create a new Deep Learning model
model = check_model()
# Check previsions and %
previsions()
