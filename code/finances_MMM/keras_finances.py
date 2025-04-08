import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras import models
from keras import layers
from keras import callbacks
import os

from plots.show_plots import ShowPlot
from plots.show_network import ShowModel


def load_dataset():
    """Load dataset, set and normalize features"""
    # Load df
    df = pd.read_csv('all_tickers_extractions_1900_01_01_2024_08_08.csv')
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], unit='s')
    # Predict only MMM Stock
    df = df[df['Stock'] == 'MMM']

    # Set X and y
    SEED = 42
    X = df.drop(['Date', 'LinReg_Pred', 'R2_Score', 'MSE', 'Avg7', 'Avg42', 'Stock'], axis=1)
    y = df['Adj Close']
    # Set train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

    # Normalize Data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Plot
    df_plot = df[df.index > 0]
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=df_plot['Date'], y=df_plot['Adj Close'], label='Trend Closing Prices', color='blue')
    sns.lineplot(x=df_plot['Date'], y=df_plot['LinReg_Pred'], label='Linear Regression', color='red')
    plt.title('Closing Prices Trend')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig(fname="trends.png")
    plt.show()

    return X_train, X_test, y_train, y_test, df


def check_model():
    """Check if a model is already created, else creates a Deep Learning model"""
    try:
        if os.path.exists(model_name):
            # Skip to Predictions
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
    This function loads the finances dataset array and creates a Deep Learning model to predict values
    Objective: predict time_series values
    """
    # Define Sequential model
    model = models.Sequential()

    # First layer
    model.add(layers.Dense(64, input_dim=X_train.shape[1], activation='relu'))
    # Second layer
    model.add(layers.Dense(32, activation='relu'))
    # Third layer as output
    model.add(layers.Dense(1, activation='linear'))

    # Show model
    ShowModel(model)

    # Compile the model and use mean squared error as loss
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    # Define EarlyStopping to prevent overfitting
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
    print('Proceeding loading the model for predictions..')


def predictions():
    # Create predictions on test data
    y_pred = model.predict(X_test)
    # Flatten y_pred to match the shape of y_test
    y_pred_flat = y_pred.flatten()

    # Check predictions with original data
    mse = np.mean((y_pred_flat - y_test) ** 2)
    print(f'Mean Squared Error on test set: {mse:.4f}')

    # Plot predictions vs actual values
    plot = ShowPlot()
    plot.lineplot(y_test.values, y_pred_flat, 'ActualVSPredicted', 'Sample', 'Price', True)

    # Make predictions for future dates
    make_predictions = input('Type 1 to make predictions: ')
    if make_predictions == '1':
        print('Predictions will be from the end of dataset until today, plus a number..')
        date_predictions = int(input('Write how many days after today you want to predict: '))
        if date_predictions <= 0:
            date_predictions = 1

        # Get the last date in the dataset
        last_date = df['Date'].max()

        # Generate future dates
        future_dates = pd.date_range(start=last_date, periods=date_predictions + 1, freq='D')[1:]

        # Use last 7 and 42 days means to create future predictions
        y_pred = model.predict(X_test)
        print(df['Adj Close'].iloc[-7:])
        print(y_pred[-7:])

        future_data_7 = [df['Adj Close'].iloc[-1]]
        future_data_42 = [df['Adj Close'].iloc[-1]]
        for i in range(1, date_predictions):
            # Calculate rolling means
            future_data_7.append(future_data_7[i-1] + y_pred[-7:].mean())
            future_data_42.append(future_data_42[i-1] + y_pred[-42:].mean())

        # Create a DataFrame for the predictions
        predictions_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price_7': future_data_7,
            'Predicted_Price_42': future_data_42
        })
        print(predictions_df.head())

        # Plot the predictions
        df_plot = df[df.index >= df.index.max() + date_predictions]
        plt.figure(figsize=(12, 6))
        plt.plot(df_plot['Date'], df_plot['Adj Close'], label='Historical Prices')
        plt.plot(predictions_df['Date'], predictions_df['Predicted_Price_7'], label='Predicted Prices mean7', color='red')
        plt.plot(predictions_df['Date'], predictions_df['Predicted_Price_42'], label='Predicted Prices mean42', color='violet')
        plt.title('Historical and Predicted Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.savefig(fname='predictions.png')
        plt.show()


# Start
model_name = 'finances.keras'

# Load values
X_train, X_test, y_train, y_test, df = load_dataset()
# Check if model is already present, else create a new Deep Learning model
model = check_model()
# Check predictions and %
predictions()