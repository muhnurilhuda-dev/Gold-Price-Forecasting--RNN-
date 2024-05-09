import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from bs4 import BeautifulSoup
import requests
import time
from csv import writer
from datetime import datetime, timedelta
import emoji
# import streamlit_authenticator as stauth 
import schedule     

print("bfhegfhfuiruhfjrwoihfrw")
st.set_page_config(page_title="nGold", page_icon=":bar_chart")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {vidibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Pilihan menu
st.sidebar.title(f"Welcome Nuril")
menu = ["Buyback", "Emas 1gr", "Emas 5gr", "Emas 10gr", "Emas 25gr", "Emas 100gr"]
choice = st.sidebar.selectbox("Menu", menu)

with open('style_css.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

if choice == "Buyback":
    # jarak_hari = st.text_input('Jarak hari yang akan diprediksi:', '')
    # st.write()
    
    # Fungsi untuk melakukan scraping data
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

    def scrape_dan_simpan():
        # Baca tanggal terakhir dari file CSV
        file_name = 'data/buy_1gr.csv'
        df5 = pd.read_csv(file_name)
        last_date_str = df5.iloc[-1]['Tanggal']
        last_date = datetime.strptime(last_date_str, '%Y-%m-%d').date()

        start_date = last_date + timedelta(days=1)
        end_date = datetime.now().date()

        if start_date <= end_date:
            # Scraping data dan menyimpannya ke dalam file CSV
            data_list = scrape_data(start_date, end_date)
            save_to_csv(data_list, file_name)

    # Load data
    df = pd.read_csv('data/buy_1gr.csv')
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df['Tanggal'] = df['Tanggal'].dt.date
    df.set_index('Tanggal',inplace=True)
    df['Harga'] = (df['Pricebuy'])
    print(df.dtypes)
    df.head(5)

    #load data lagi
    df2 = pd.read_csv('data/buy_1gr.csv')
    df2['Tanggal'] = pd.to_datetime(df2['Tanggal'])
    df2.set_index('Tanggal', inplace=True)
    print(df2.dtypes)
    df2.head(5)

    data_set = df2.iloc[:, 0:1] #.values
    pd.set_option('display.max_columns', None)

    data_set.head(5)
    
    # Learning / Preprocessing data
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0,1))
    data_set_scaled = sc.fit_transform(data_set)
    print(data_set_scaled)

    # Multiple features from data provided to the model
    X = []
    backcandles = 120 # Jumlah hari mundur / kebelakang
    print(data_set_scaled.shape)
    for j in range(1): # jumlah kolom = 8
      X.append([])
      for i in range(backcandles, data_set_scaled.shape[0]):
        X[j].append(data_set_scaled[i - backcandles:i, j])

    X = np.moveaxis(X, [0], [2])

    # -1 untuk memilih kolom terakhir
    X, yi = np.array(X), np.array(data_set_scaled[backcandles:, -1])
    y=np.reshape(yi,(len(yi),1))

    # Split data into train test sets
    splitlimit = int(len(X)*0.8)
    print(splitlimit)

    X_train, X_test = X[:splitlimit], X[splitlimit:]
    y_train, y_test = y[:splitlimit], y[splitlimit:]
    print(X_train.shape)
    print(X_test.shape)

    from keras.models import Sequential

    import tensorflow as tf
    import keras
    from keras import optimizers
    from keras.callbacks import History
    from keras.models import Model
    from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate, TimeDistributed
    import numpy as np
    import matplotlib.pyplot as plt

    # backcandles = 30

    lstm_input = Input(shape=(backcandles, 1), name="lstm_input")
    inputs = LSTM(150, name="first_layer")(lstm_input)
    inputs = Dense(1, name="dense_layer")(inputs)
    output = Activation('linear', name="output")(inputs)
    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam()
    model.compile(optimizer=adam, loss='mse')

    print("X_train shape:", X_train.shape)
    print("X_train dtype:", X_train.dtype)
    print("y_train shape:", y_train.shape)
    print("y_train dtype:", y_train.dtype)

    model_test = model.fit(x=X_train, y=y_train, batch_size=15, epochs=30, shuffle=True, validation_split=0.1)

    plt.plot(model_test.history['loss'])
    plt.plot(model_test.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    # Prediction
    y_pred = model.predict(X_test)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # MSE
    mse = mean_squared_error(y_test, y_pred)
    print("MSE score: ", mse)

    # RMSE
    rmse = np.sqrt(mse)
    print("RMSE score: ", rmse)

    # MAE
    mae = mean_absolute_error(y_test, y_pred)
    print("MAE score: ", mae)

    #R2
    r2 = r2_score(y_test, y_pred)
    print("r2_score: ", r2)

    # plt.figure(figsize=(16,8))
    # plt.plot(y_test, color="red", label="Test")
    # plt.plot(y_pred, color="blue", label="Prediction")
    # plt.legend()
    # plt.show()

    # Get the last backcandles days' price
    last_backandles_prices = data_set_scaled[-backcandles:]

    # Reshape the data to match the modul input shape
    last_backandles_prices = last_backandles_prices.reshape((1, backcandles, 1))

    # Predict tomorrow's price
    predicted_tomorrow_price = model.predict(last_backandles_prices)

    # Inverse transform the predicted price to get the actual price
    predicted_tomorrow_price = sc.inverse_transform(predicted_tomorrow_price)

    print("Tomorrow's price: ", predicted_tomorrow_price[0][0])

    # grafik sebelum input
    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df.index, y=df['Pricebuy'], mode='lines', name='Harga Emas Aktual'))

    # garis grafik harga emas aktual
    fig.add_trace(go.Scatter(x=df.index, y=df['Pricebuy'], mode='lines', name='Harga Emas Aktual'))
    # fig.add_trace(go.Scatter(x=df.index, y=wmavg, mode='lines', name='Prediksi Harga Emas'))

    # garis grafik perkiraan harga emas
    # predicted_dates = pd.date_range(start=df.index[-1], periods=len(y_pred) + 1)[1:] # men-generate tanggal untuk prediksi
    # actual_predicted_prices = sc.inverse_transform(predicted_dates)
    # fig.add_trace(go.Scatter(x=predicted_dates, y=y_pred.flatten(), mode='lines', name='Prediksi Harga Emas'))

    # Prediksi harga besok
    # fig.add_trace(go.Scatter(x=[df.index[-1], df.index[-1] + timedelta(days=int(days_input))], y=[df['Pricebuy'].iloc[-1], predicted_tomorrow_price[0][0]], mode='markers+lines', name="Prediksi Harga Esok"))

    # layout grafik
    fig.update_layout(
        xaxis_title='Tanggal',
        yaxis_title='Harga Emas (Rupiah)',
            xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label='1 w', step='day', stepmode='backward'),
                    dict(count=3, label='3 m', step='month', stepmode='backward'),
                    dict(count=6, label='6 m', step='month', stepmode='backward'),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(visible=True),
            type='date'
        ),
        width=750,  # Lebar grafik (dalam piksel)
        height=400
    )

    st.plotly_chart(fig)
    
    #####################################################################################
    
    days_input = st.text_input("Masukkan jumlah hari kedepan: ", "")

    if st.button("Show prediction"):
        if days_input == "":
            st.warning("Please fill the input field first!")
        else:
            num_days = int(days_input)
            # grafik setelah input
            fig = go.Figure()
            # fig.add_trace(go.Scatter(x=df.index, y=df['Pricebuy'], mode='lines', name='Harga Emas Aktual'))

            # garis grafik harga emas aktual
            fig.add_trace(go.Scatter(x=df.index, y=df['Pricebuy'], mode='lines', name='Harga Emas Aktual'))
            # fig.add_trace(go.Scatter(x=df.index, y=wmavg, mode='lines', name='Prediksi Harga Emas'))

            # garis grafik perkiraan harga emas
            # actual_predicted_prices = sc.inverse_transform(predicted_dates)
            
            # predicted_dates = pd.date_range(start=df.index[-1], periods=len(y_pred) + 1)[1:] # men-generate tanggal untuk prediksi
            # fig.add_trace(go.Scatter(x=predicted_dates, y=y_pred.flatten(), mode='lines', name='Prediksi Harga Emas'))

            # Prediksi harga besok
            # fig.add_trace(go.Scatter(x=[df.index[-1], df.index[-1] + timedelta(days=15)], y=[df['Pricebuy'].iloc[-1], predicted_tomorrow_price[0][0]], mode='markers+lines', name="Prediksi Harga Esok"))
            fig.add_trace(go.Scatter(x=[df.index[-1], df.index[-1] + timedelta(days=int(days_input))], y=[df['Pricebuy'].iloc[-1], predicted_tomorrow_price[0][0]], mode='markers+lines', name="Prediksi Harga Esok"))

            # layout grafik
            fig.update_layout(
                xaxis_title='Tanggal',
                yaxis_title='Harga Emas (Rupiah)',
                    xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=7, label='1 w', step='day', stepmode='backward'),
                            dict(count=3, label='3 m', step='month', stepmode='backward'),
                            dict(count=6, label='6 m', step='month', stepmode='backward'),
                            dict(step='all')
                        ])
                    ),
                    rangeslider=dict(visible=True),
                    type='date'
                ),
                width=750,  # Lebar grafik (dalam piksel)
                height=400
            )

            st.plotly_chart(fig)

            # col1, col2 = st.columns(2)
            # with col1:


            # st.button('Re-train', on_click=scrape_dan_simpan)