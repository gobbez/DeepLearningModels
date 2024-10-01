import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


def workon_finances():
    """
    This function loads the finances dataset array and creates a Deep Learning model to predict values
    Objective: predict time_series values
    """
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

    # Check if a model is already created
    check_model = 0
    select_name_model = 'finances.h5'
    try:
        model = load_model('finances.h5')
        select_new = input('Type 1 if you want to create a new model instead: ')
        if select_new == '1':
            select_name_model = input('Type the name of the new model: ')
            check_model = 0
        else:
            check_model = 1
    except:
        print('No model loaded. Proceed to creating, training and saving a new one')

    if check_model == 0:
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
        plt.show()

        # DataFrame analyzed, it's ready to continue

        # Define Sequential model
        model = Sequential()

        # First layer
        model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
        # Second layer
        model.add(Dense(32, activation='relu'))
        # Third layer as output
        model.add(Dense(1, activation='linear'))

        # Compile the model and use mean squared error as loss
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

        # Define EarlyStopping to prevent overfitting
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

        # Mean Absolute Error
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mse'], label='Train MAE')
        plt.plot(history.history['val_mse'], label='Validation MAE')
        plt.title('Mean Absolute Error')
        plt.xlabel('Epochs')
        plt.ylabel('MAE')
        plt.legend()

        plt.show()

        # Create predictions on test data
        y_pred = model.predict(X_test)
        # Flatten y_pred to match the shape of y_test
        y_pred_flat = y_pred.flatten()

        # Check predictions with original data
        mse = np.mean((y_pred_flat - y_test) ** 2)
        print(f'Mean Squared Error on test set: {mse:.4f}')

        # Plot predictions vs actual values
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values, label='Actual', color='blue')
        plt.plot(y_pred_flat, label='Predicted', color='red')
        plt.title('Actual vs Predicted')
        plt.xlabel('Sample')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

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
        plt.show()



workon_finances()