from turtle import mode
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import requests
from bs4 import BeautifulSoup
from csv import writer

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
    df = pd.read_csv('data/buy_1gr.csv')
    st.write(df)

# Scrape Data page
elif choice == "Scrape Data":
    st.title("Scrape Gold Price Data")

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
        for date in pd.date_range(start_date, end_date):
            month_name_id = bulan[date.month]
            url_day = f"https://harga-emas.org/history-harga/{date.year}/{month_name_id}/{date.day}/"
            page = requests.get(url_day)
            soup = BeautifulSoup(page.content, 'html.parser')
            lists = soup.find('div', class_='col-md-8')
            row_data = [date.strftime('%Y-%m-%d')]
            index = 0
            for item in lists.findAll('tr'):
                index += 1
                if index == 21:
                    base_value = item.findAll('b')
                    index_core = 0
                    for core in base_value:
                        index_core += 1
                        if index_core == 2:
                            value = core.text.split('+')[0].split('-')[0].split('(')[0]
                            value = value.replace('.', '').strip()
                            value = value.replace('Rp', '').strip()
                            value = value.replace('/', '').strip()
                            value = value.replace('gram', '').strip()
                            row_data.append(value)
            data_list.append(row_data)
            time.sleep(1)  # Jeda untuk mencegah terlalu banyak permintaan ke website
        return data_list

    def save_to_csv(data_list, file_name):
        with open(file_name, 'a', newline='') as file:
            csv_writer = writer(file)
            csv_writer.writerows(data_list)

    start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=30))
    end_date = st.date_input("End Date", datetime.now().date())

    if st.button("Scrape Data"):
        data_list = scrape_data(start_date, end_date)
        save_to_csv(data_list, 'data/buy_1gr.csv')
        st.success("Data scraped and saved successfully!")

# Forecast page
elif choice == "Forecast":
    st.title("Gold Price Forecast")

    df = pd.read_csv('data/buy_1gr.csv')
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df.set_index('Tanggal', inplace=True)
    df['Pricebuy'] = df['Pricebuy'].astype(str).str.replace('.', '').astype(float)

    # Prepare data for LSTM
    data = df['Pricebuy'].values
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

    look_back = st.slider("Look back", 1, 60, 30)
    trainX, trainY = create_dataset(train_data, look_back)
    testX, testY = create_dataset(test_data, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    # LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(trainX, trainY, batch_size=1, epochs=1)

    # Predictions
    train_predict = model.predict(trainX)
    test_predict = model.predict(testX)

    train_predict = scaler.inverse_transform(train_predict)
    trainY = scaler.inverse_transform([trainY])
    test_predict = scaler.inverse_transform(test_predict)
    testY = scaler.inverse_transform([testY])

    # Plotting
    train_predict_plot = np.empty_like(data)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

    test_predict_plot = np.empty_like(data)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (look_back * 2) + 1:len(data) - 1, :] = test_predict

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Pricebuy'], mode='lines', name='Harga Emas Aktual'))
    fig.add_trace(go.Scatter(x=df.index, y=train_predict_plot[:, 0], mode='lines', name='Train Predict'))
    fig.add_trace(go.Scatter(x=df.index, y=test_predict_plot[:, 0], mode='lines', name='Test Predict'))
    fig.update_layout(
        xaxis_title='Tanggal',
        yaxis_title='Harga Emas (Rupiah)',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label='1w', step='day', stepmode='backward'),
                    dict(count=1, label='1m', step='month', stepmode='backward'),
                    dict(count=6, label='6m', step='month', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(visible=True),
            type='date'
        )
    )
    st.plotly_chart(fig)
    
    # User input for number of days to predict
    num_days = st.number_input("Number of days to predict: ", min_value=1, max_value=30, value=1)
    
    # if st.button("Predict"):
    last_prices = scaled_data[-look_back:]
    predicted_prices = []
        
    for _ in range(num_days):
        last_prices_reshaped = last_prices.reshape((1, look_back, 1))
        predicted_price = model.predict(last_prices_reshaped)
        predicted_prices.append(predicted_price[0][0])
        last_prices = np.append(last_prices[1:], predicted_price, axis=0)
            
    predicted_prices = np.array(predicted_prices).reshape(-1, 1)
    predicted_prices = scaler.inverse_transform(predicted_prices)
        
    # Create future dates
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, num_days + 1)]
        
    # Plot future predictions
    fig.add_trace(go.Scatter(x=future_dates, y=predicted_prices[:, 0], mode='lines+markers', name='Future Predict'))
    st.plotly_chart(fig)
        
    # Show predicted prices
    predicted_df = pd.DataFrame(predicted_prices, index=future_dates, columns=['Predicted Price'])
    st.write(predicted_df)