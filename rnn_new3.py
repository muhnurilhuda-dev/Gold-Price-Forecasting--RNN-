# Imports and initial setup
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import os
import locale
import time
import requests
from bs4 import BeautifulSoup
from csv import writer

# Streamlit configuration
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
    df['Harga 1gr'] = df['Price1']
    df['Harga 2gr'] = df['Price2']
    df['Harga 3gr'] = df['Price3']
    df['Harga 5gr'] = df['Price5']
    df['Harga 10gr'] = df['Price10']
    df['Harga 25gr'] = df['Price25']
    df['Harga 50gr'] = df['Price50']
    df['Harga 100gr'] = df['Price100']
    df.drop(['Price1', 'Price2', 'Price3', 'Price5', 'Price10', 'Price25', 'Price50', 'Price100'], axis=1, inplace=True)
    st.write(df)

# Scrape Data page
elif choice == "Scrape Data":
    st.title("Scrape Gold Price Data")
    with st.status("Scraping data...", expanded=True) as status:
        def scrape_data(start_date, end_date, existing_dates):
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
                
                # Checking the price data is already existed or not
                date_str = date.strftime("%Y-%m-%d")
                if date_str in existing_dates:
                    continue
                
                month_name_id = bulan[date.month]
                url_day = f"https://harga-emas.org/history-harga/{date.year}/{month_name_id}/{date.day}/"                
                
                try:
                    page = requests.get(url_day)
                    soup = BeautifulSoup(page.content, 'html.parser')
                    lists = soup.find('div', class_='col-md-8')
                    row_data = [date.strftime('%Y-%m-%d')]
                    index = 0
                    for item in lists.findAll('tr'):
                        index += 1
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

        start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=30), min_value=date(2014,1,1), max_value=datetime.now(), format="DD/MM/YYYY")
        end_date = st.date_input("End Date", datetime.now().date(), min_value=date(2014,1,1), max_value=datetime.now(), format="DD/MM/YYYY")
        
        # Load existing price data
        existing_data = pd.read_csv('data/harga_emas_new2.csv')
        existing_dates = set(existing_data['Tanggal'])

        if st.button("Scrape Data"):
            data_list = scrape_data(start_date, end_date, existing_dates)
            save_to_csv(data_list, 'data/harga_emas_new2.csv')
            status.update(label="Getting data complete!", state="complete", expanded=True)
            st.success("Data scraped and saved successfully!")

# Forecast page
elif choice == "Forecast":
    st.title("Gold Price Forecast")

    df = pd.read_csv('data/harga_emas_new2.csv')
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df.set_index('Tanggal', inplace=True)

    df['Harga'] = df['Price1'].astype(str).str.replace('.', '').astype(float)
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

    look_back = 30
    trainX, trainY = create_dataset(train_data, look_back)
    testX, testY = create_dataset(test_data, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    model_path = 'saved_model/gold_price_lstm.h5'
    
    retrain = st.button("Retrain Model")
    if retrain or not os.path.exists(model_path):
        model = Sequential()
        model.add(LSTM(300, return_sequences=True, input_shape=(look_back, 1)))
        model.add(LSTM(150, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        learning_rate = 0.001
        batch_size = 30
        epochs = 30

        adam = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=adam, loss='mean_squared_error')
        model.fit(x=trainX, y=trainY, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.1)

        model.save(model_path)
        st.success("Model trained and saved")
    else:
        model = load_model(model_path)
        st.success("Loaded model from disk")
    
    # Predictions
    train_predict = model.predict(trainX)
    test_predict = model.predict(testX)

    train_predict = scaler.inverse_transform(train_predict)
    trainY = scaler.inverse_transform(trainY.reshape(-1, 1))
    test_predict = scaler.inverse_transform(test_predict)
    testY = scaler.inverse_transform(testY.reshape(-1, 1))
    
    # Calculate accuracy metrics
    train_mse = mean_squared_error(trainY, train_predict)
    train_mae = mean_absolute_error(trainY, train_predict)
    train_mape = mean_absolute_percentage_error(trainY, train_predict)
    test_mse = mean_squared_error(testY, test_predict)
    test_mae = mean_absolute_error(testY, test_predict)
    test_mape = mean_absolute_percentage_error(testY, test_predict)

    st.write("Train MSE:", train_mse)
    st.write("Train MAE:", train_mae)
    st.write("Train MAPE:", train_mape)
    st.write("Test MSE:", test_mse)
    st.write("Test MAE:", test_mae)
    st.write("Test MAPE:", test_mape)

    # Plotting
    st.subheader('Actual vs Predicted Prices')
    trace1 = go.Scatter(
        x=df.index[:train_size],
        y=trainY.flatten(),
        mode='lines',
        name='Actual Train Prices'
    )
    trace2 = go.Scatter(
        x=df.index[:train_size],
        y=train_predict.flatten(),
        mode='lines',
        name='Predicted Train Prices'
    )
    trace3 = go.Scatter(
        x=df.index[train_size:],
        y=testY.flatten(),
        mode='lines',
        name='Actual Test Prices'
    )
    trace4 = go.Scatter(
        x=df.index[train_size:],
        y=test_predict.flatten(),
        mode='lines',
        name='Predicted Test Prices'
    )

    data = [trace1, trace2, trace3, trace4]
    layout = go.Layout(
        title='Gold Prices: Actual vs Predicted',
        xaxis={'title': 'Date'},
        yaxis={'title': 'Price'},
        hovermode='closest'
    )
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig)
    
    # Weighted Moving Average (WMA) Calculation
    def wma(values, period):
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights[::-1], 'valid') / weights.sum()
    
    period = 7
    wma_values = wma(df['Harga'].values, period)
    
    # Adding NaN values to align the WMA with the original data length
    wma_full = np.empty(len(df['Harga']))
    wma_full[:period-1] = np.nan
    wma_full[period-1:] = wma_values
    
    df['WMA'] = wma_full

    train_wma = df['WMA'].iloc[:train_size].dropna()
    test_wma = df['WMA'].iloc[train_size:].dropna()
    
    # Ensure the lengths of train_wma and trainY match
    valid_length = min(len(train_wma), len(trainY))
    train_wma = train_wma[-valid_length:]
    trainY_wma = trainY[-valid_length:]
    
    # Ensure the lengths of test_wma and testY match
    valid_length = min(len(test_wma), len(testY))
    test_wma = test_wma[-valid_length:]
    testY_wma = testY[-valid_length:]
    
    train_wma_mse = mean_squared_error(trainY_wma, train_wma)
    train_wma_mae = mean_absolute_error(trainY_wma, train_wma)
    train_wma_mape = mean_absolute_percentage_error(trainY_wma, train_wma)
    test_wma_mse = mean_squared_error(testY_wma, test_wma)
    test_wma_mae = mean_absolute_error(testY_wma, test_wma)
    test_wma_mape = mean_absolute_percentage_error(testY_wma, test_wma)

    st.write("Train WMA MSE:", train_wma_mse)
    st.write("Train WMA MAE:", train_wma_mae)
    st.write("Train WMA MAPE:", train_wma_mape)
    st.write("Test WMA MSE:", test_wma_mse)
    st.write("Test WMA MAE:", test_wma_mae)
    st.write("Test WMA MAPE:", test_wma_mape)

    st.subheader('Actual vs WMA Prices')
    trace1 = go.Scatter(
        x=df.index[:train_size][-len(train_wma):],
        y=trainY_wma.flatten(),
        mode='lines',
        name='Actual Train Prices'
    )
    trace2 = go.Scatter(
        x=df.index[:train_size][-len(train_wma):],
        y=train_wma,
        mode='lines',
        name='WMA Train Prices'
    )
    trace3 = go.Scatter(
        x=df.index[train_size:][-len(test_wma):],
        y=testY_wma.flatten(),
        mode='lines',
        name='Actual Test Prices'
    )
    trace4 = go.Scatter(
        x=df.index[train_size:][-len(test_wma):],
        y=test_wma,
        mode='lines',
        name='WMA Test Prices'
    )

    data = [trace1, trace2, trace3, trace4]
    layout = go.Layout(
        title='Gold Prices: Actual vs WMA',
        xaxis={'title': 'Date'},
        yaxis={'title': 'Price'},
        hovermode='closest'
    )
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig)
