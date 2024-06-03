from random import shuffle
from turtle import width
from sklearn.metrics import mean_absolute_percentage_error
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup
from csv import writer
import os

st.set_page_config(page_title="Gold Price Forecasting", page_icon=":bar_chart:")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Sidebar menu
st.sidebar.title("Gold Price Forecasting")
menu = ["Home", "Data", "Scrape Data", "Forecast"]
choice = st.sidebar.selectbox("Menu", menu)

# Home page
if choice == "Home":
    st.title("Gold Price Forecasting")
    st.write("Welcome to the Gold Price Forecasting app!")

# Data page
elif choice == "Data":
    st.title("Gold Price Data")
    df = pd.read_csv('data/harga_emas_new2.csv')
    st.write(df)

# Scrape Data page
elif choice == "Scrape Data":
    st.title("Scrape Gold Price Data")
    with st.status("Scraping data...", expanded=True) as status:
        def scrape_data(start_date, end_date):
            bulan = {
                1: "Januari",
                2: "Februari",
                3: "Maret",
                4: "April",
                5: "Mei",
                6: "Juni",
                7: "Juli",
                8: "Agustus",
                9: "September",
                10: "Oktober",
                11: "November",
                12: "Desember"
            }
            data_list = []
            st.write("Searching for data...")
            time.sleep(2)                        
            st.write("URL found...")
            time.sleep(1)         
            st.write("Getting the data form URL...")       

            for date in pd.date_range(start_date, end_date):
                month_name_id = bulan[date.month]
                url_day = f"https://harga-emas.org/history-harga/{date.year}/{month_name_id}/{date.day}/"                
                
                # max_entries = 3
                # for attempt in range(max_entries):
                #     try:
                #         page = requests.get(url_day, timeout=10)
                #         page.raise_for_status() # Raise an exception for HTTP requests
                #         break
                #     except requests.exceptions.RequestException as e:
                #         st.write(f"Attempt {attempt + 1} failed: {e}")
                #         time.sleep(2)
                # else:
                #     st.error("Failed to retrieve data after multiple attempts.")
                #     return []
                
                try:
                    page = requests.get(url_day)
                    soup = BeautifulSoup(page.content, 'html.parser')
                    lists = soup.find('div', class_='col-md-8')
                    row_data = [date.strftime('%Y-%m-%d')]
                    index = 0
                    for item in lists.findAll('tr'):
                        index += 1
                        # if index == 21:
                        if 11 < index < 20:
                            base_value = item.findAll('td')
                            index_core = 0
                            for core in base_value:
                                index_core += 1
                                if index_core == 2:
                                    value = core.text.split('+')[0].split('-')[0].split('(')[0]
                                    value = value.replace('.', '').strip()
                                    row_data.append(value)
                    data_list.append(row_data)
                    time.sleep(1)  # Jeda untuk mencegah terlalu banyak permintaan ke website
                except requests.exceptions.RequestException as e:
                    st.error(f"Error fetching data for {date.strftime('%Y-%m-%d')}: {e}")
                    continue
            return data_list

        def save_to_csv(data_list, file_name):
            with open(file_name, 'a', newline='') as file:
                csv_writer = writer(file)
                csv_writer.writerows(data_list)

        start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=30))
        end_date = st.date_input("End Date", datetime.now().date())

        if st.button("Scrape Data"):
            data_list = scrape_data(start_date, end_date)
            # save_to_csv(data_list, 'data/buy_1gr.csv')
            save_to_csv(data_list, 'data/harga_emas_new2.csv')
            status.update(label="Getting data complete!", state="complete", expanded=True)
            st.success("Data scraped and saved successfully!")

# Forecast page
elif choice == "Forecast":
    st.title("Gold Price Forecast")

    # df = pd.read_csv('data/buy_1gr.csv')
    df = pd.read_csv('data/harga_emas_new2.csv')
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df.set_index('Tanggal', inplace=True)

    # Ensure all values in Price1 are strings
    df['Harga'] = df['Price1'].astype(str).str.replace('.', '').astype(float)
    # df['Harga'] = (df['Price1'])
    df.drop(['Price1', 'Price2', 'Price3', 'Price5', 'Price10', 'Price25', 'Price50', 'Price100'], axis=1, inplace=True)

    # Prepare data for LSTM
    data = df['Harga'].values
    data = data.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    # look_back = st.slider("Look back", 1, 60, 30)
    look_back = 30
    trainX, trainY = create_dataset(train_data, look_back)
    testX, testY = create_dataset(test_data, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    model_path = 'saved_model/gold_price_lstm.h5'
    
    retrain = st.button("Retrain Model")
    if retrain or not os.path.exists(model_path):
        # LSTM model
        model = Sequential()
        model.add(LSTM(300, return_sequences=True, input_shape=(look_back, 1)))
        model.add(LSTM(150, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        # Hyperparameters to tune
        lstm_units = 150  # Start with 150, tune this value
        learning_rate = 0.001  # Start with 0.001, tune this value
        batch_size = 30  # Start with 30, tune this value
        epochs = 30  # Start with 20, tune this value
        dropout_rate = 0.2  # Start with 0.2, tune this value

        adam = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=adam, loss='mean_squared_error')
        model.fit(x=trainX, y=trainY, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.1)

        # Save the model
        model.save(model_path)
        st.success("Model trained and saved")
    else:
        model = load_model(model_path)
        st.success("Loaded model from disk")
    
    # if os.path.exists(model_path):
    #     model = load_model(model_path)
    #     st.write("Model loaded")
    # else:
        
    #     # LSTM model
    #     model = Sequential()
    #     model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    #     model.add(LSTM(50, return_sequences=False))
    #     model.add(Dense(25))
    #     model.add(Dense(1))
    
    #     # Hyperparameters to tune
    #     backcandles = 30  # Assuming 30 time steps
    #     lstm_units = 150  # Start with 150, tune this value
    #     learning_rate = 0.001  # Start with 0.001, tune this value
    #     batch_size = 30  # Start with 15, tune this value
    #     epochs = 20  # Start with 30, tune this value
    #     dropout_rate = 0.2  # Start with 0.2, tune this value
    
    #     adam = optimizers.Adam(learning_rate=learning_rate)
    #     model.compile(optimizer=adam, loss='mean_squared_error')
        
        

    #     # model.compile(optimizer='adam', loss='mean_squared_error')
    #     model.fit(x=trainX, y=trainY, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.1)  # Increased epochs for better training
    
    #     # Save the trained model
    #     model.save(model_path)
    #     st.write("Model trained and saved")

    # Predictions
    train_predict = model.predict(trainX)
    test_predict = model.predict(testX)

    train_predict = scaler.inverse_transform(train_predict)
    # trainY = scaler.inverse_transform([trainY])
    trainY = scaler.inverse_transform(trainY.reshape(-1, 1))
    test_predict = scaler.inverse_transform(test_predict)
    # testY = scaler.inverse_transform([testY])
    testY = scaler.inverse_transform(testY.reshape(-1, 1))
    
    # Calculate accuracy metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # train_mse = mean_squared_error(trainY[0], train_predict[:, 0])
    # train_mae = mean_absolute_error(trainY[0], train_predict[:, 0])
    # train_mape = mean_absolute_percentage_error(trainY[0], train_predict[:, 0])
    # test_mse = mean_squared_error(testY[0], test_predict[:, 0])
    # test_mae = mean_absolute_error(testY[0], test_predict[:, 0])
    
    train_mse = mean_squared_error(trainY, train_predict)
    train_mae = mean_absolute_error(trainY, train_predict)
    train_mape = mean_absolute_percentage_error(trainY, train_predict)
    test_mse = mean_squared_error(testY, test_predict)
    test_mae = mean_absolute_error(testY, test_predict)
    test_mape = mean_absolute_percentage_error(testY, test_predict)
    
    # def mean_absolute_percentage_error(testY, test_predict):
    #     return np.mean(np.abs((testY, test_predict) / test_predict)) * 100
    
    
    st.write(f"Train MSE: {train_mse:.4f}")
    st.write(f"Train MAE: {train_mae:.4f}")
    st.write(f"Train MAPE: {train_mape * 100}%")
    st.write(f"Test MSE: {test_mse:.4f}")
    st.write(f"Test MAE: {test_mae:.4f}")
    st.write(f"Test MAPE: {test_mape * 100}%")

    # Plotting
    train_predict_plot = np.empty_like(data)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

    test_predict_plot = np.empty_like(data)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(data) - 1, :] = test_predict

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Harga'], mode='lines', name='Harga Emas Aktual'))
    fig.add_trace(go.Scatter(x=df.index, y=train_predict_plot[:, 0], mode='lines', name='Train Predict'))
    fig.add_trace(go.Scatter(x=df.index, y=test_predict_plot[:, 0], mode='lines', name='Test Predict'))
    fig.update_layout(
        xaxis_title='Tanggal',
        yaxis_title='Harga Emas (Rupiah)',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label='1w', step='day', stepmode='backward'),
                    dict(count=3, label='3m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(visible=True),
            type='date'
        ),
        width=750,
        height=400
    )
    st.plotly_chart(fig)

    # User input for number of days to predict
    num_days = st.number_input("Number of days to predict", min_value=1, max_value=30, value=1)

    # if st.button("Predict"):
    
    # predicted_tomorrow_prices = []
    
    # for _ in range(num_days):
    #     # Get the last backcandles days' price
    #     last_backcandles_prices = scaled_data[-look_back:]
    
    #     # Reshape the data to match the modul input shape
    #     last_backcandles_prices = last_backcandles_prices.reshape((1, look_back, 1))
    
    #     # Predict tomorrow's price
    #     predicted_tomorrow_price = model.predict(last_backcandles_prices)
    #     predicted_tomorrow_prices.append(predicted_tomorrow_price[0][0])
    
    #     # Inverse transform the predicted price to get the actual price
    #     predicted_tomorrow_price = scaler.inverse_transform(predicted_tomorrow_price)
        
    
    # fig.add_trace(go.Scatter(x=[df.index[-1], df.index[-1] + timedelta(days=int(num_days))], y=[df['Harga'].iloc[-1], predicted_tomorrow_price[:, 0]], mode='markers+lines', name="Prediksi Harga Esok"))
    
    
    
    # START PREDICTION
    last_prices = scaled_data[-look_back:]
    predicted_prices = []

    for _ in range(num_days):
        last_prices_reshaped = last_prices.reshape((1, look_back, 1))
        predicted_price = model.predict(last_prices_reshaped)
        predicted_prices.append(predicted_price[0][0])
        last_prices = np.append(last_prices[1:], predicted_price, axis=0)
        
    predicted_prices = np.array(predicted_prices).reshape(-1, 1)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # # Create future dates
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, num_days + 1)]

    # Plot future predictions
    fig.add_trace(go.Scatter(x=future_dates, y=predicted_prices[:, 0], mode='lines+markers', name='Future Predict'))    
    st.plotly_chart(fig)
    
    
    # # CHECKING THE ACCURACY
    # from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
    # # MSE
    # mse = mean_squared_error()
    

    # Show predicted prices
    # predicted_df = pd.DataFrame(predicted_prices, index=future_dates, columns=['Predicted Price'])
    # st.write(predicted_df)
