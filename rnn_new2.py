from calendar import month_name
from random import shuffle
from turtle import width
from click import group
from sklearn.metrics import mean_absolute_percentage_error
from sqlalchemy import values
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
import time
import requests
from bs4 import BeautifulSoup
from csv import writer
import os
import locale

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

        start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=30), min_value=date(2014,1,1), max_value=datetime.now(), format="DD/MM/YYYY")
        end_date = st.date_input("End Date", datetime.now().date(), min_value=date(2014,1,1), max_value=datetime.now(), format="DD/MM/YYYY")
        
        # Load existing price data
        existing_data = pd.read_csv('data/harga_emas_new2.csv')
        existing_dates = set(existing_data['Tanggal'])

        if st.button("Scrape Data"):
            data_list = scrape_data(start_date, end_date, existing_dates)
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

    model_path = 'saved_model/model.h5'
    
    retrain = st.button("Retrain Model")
    if retrain or not os.path.exists(model_path):
        # LSTM model
        model = Sequential()
        model.add(LSTM(300, return_sequences=True, input_shape=(look_back, 1)))
        model.add(LSTM(150, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        # Konfigurasi Hyperparameter
        learning_rate = 0.001
        batch_size = 20 
        epochs = 30
        dropout_rate = 0.2 

        adam = optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=adam, loss='mean_squared_error')
        model.fit(x=trainX, y=trainY, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.1)

        # Save the model
        model.save(model_path)
        st.success("Model trained and saved successfully")
    else:
        model = load_model(model_path)
        st.success("Model is loaded successfully")

    # Predictions
    train_predict = model.predict(trainX)
    test_predict = model.predict(testX)

    train_predict = scaler.inverse_transform(train_predict)
    # trainY = scaler.inverse_transform([trainY])
    trainY = scaler.inverse_transform(trainY.reshape(-1, 1))
    test_predict = scaler.inverse_transform(test_predict)
    # testY = scaler.inverse_transform([testY])
    testY = scaler.inverse_transform(testY.reshape(-1, 1))
    
    # # Menghitung tingkat error
    # from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # train_mse = mean_squared_error(trainY, train_predict)
    # train_mape = mean_absolute_percentage_error(trainY, train_predict)
    # test_mse = mean_squared_error(testY, test_predict)
    # test_mape = mean_absolute_percentage_error(testY, test_predict)
    
    # # def mean_absolute_percentage_error(testY, test_predict):
    # #     return np.mean(np.abs((testY, test_predict) / test_predict)) * 100
    
    
    # st.write(f"Train MSE: {train_mse:.4f}")
    # st.write(f"Train MAPE: {train_mape * 100}%")
    # st.write(f"Test MSE: {test_mse:.4f}")
    # st.write(f"Test MAPE: {test_mape * 100}%")

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
    # st.plotly_chart(fig)

    # Jumlah hari kedepan yang akan diprediksi
    num_days = st.number_input("Number of days to predict", min_value=1, max_value=30, value=1)
    
    # PREDICTION
    last_prices = scaled_data[-look_back:]
    predicted_prices = []

    for _ in range(num_days):
        last_prices_reshaped = last_prices.reshape((1, look_back, 1))
        predicted_price = model.predict(last_prices_reshaped)
        predicted_prices.append(predicted_price[0][0])
        last_prices = np.append(last_prices[1:], predicted_price, axis=0)
        
    predicted_prices = np.array(predicted_prices).reshape(-1, 1)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Membuat tanggal kedepan
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, num_days + 1)]

    # Plotting prediksi
    fig.add_trace(go.Scatter(x=future_dates, y=predicted_prices[:, 0], mode='lines+markers', name='Future Predict'))    
    st.plotly_chart(fig)
    
    # Menampilkan data dalam bentuk tabel
    recent_data = df[-7:]
    recent_data_scaled = scaler.transform(recent_data)
    
    # Membuat dataframe untuk data harga 7 hari terakhir dan prediksi untuk besok   
    recent_df = pd.DataFrame({
        'Tanggal': recent_data.index,
        'Harga': recent_data['Harga'].values,
        # 'Prediksi': np.append(recent_prices[1:], predicted_prices[-1])
    })
    
    st.write("Data harga emas untuk 7 hari terakhir dan prediksi besok:")
    st.table(recent_df)
    predicted_tomorrow_price = float(predicted_prices[0][0])
    predicted_tomorrow_currency = locale.currency(predicted_tomorrow_price, grouping=True)
    st.write(f"Prediksi harga besok: {predicted_tomorrow_currency}")
    
    # SELLING GOLD RECOMMENDATION (Rekomendasi Jual Emas)
    st.header("Rekomendasi Jual Emas")
    purchase_date = st.date_input("Pilih tanggal saat Anda membeli emas", min_value=date(2014,1,1), max_value=datetime.now(), format="DD/MM/YYYY")
    # purchase_price = st.number_input("Masukkan satuan gram emas yang ingin anda jual")
    purchase_date_str = purchase_date.strftime("%Y-%m-%d")
    
    locale.setlocale(locale.LC_ALL, '')
    
    if purchase_date_str in df.index:
        purchase_price = float(df.loc[purchase_date_str]['Harga'])
        currency_price = locale.currency(purchase_price, grouping=True)
        st.write(f"Harga aktual pada tanggal {purchase_date_str}: {currency_price}/gram")
    else:
        st.error("Data harga pada tanggal tersebut kosong.")
    
    
    # def tombol_rekomendasi():
    #     st.button("Tampilkan Rekomendasi", on_click=tampilkan_rekomendasi)
    
    # if st.button("Tampilkan Rekomendasi"):
    with st.container():
        # rekomendasi = st.button("Tampilkan Rekomendasi")
        def tampilkan_rekomendasi():
        # if rekomendasi:
            with st.spinner("Loading..."):
            
            # if purchase_price in df.index:
                # purchase_price = float(purchase_price)
                tomorrow_price = float(predicted_prices)
                tomorrow_currency_price = locale.currency(tomorrow_price, grouping=True)
                if tomorrow_price > purchase_price:
                    profit_currency = locale.currency((tomorrow_price - purchase_price), grouping=True)
                    time.sleep(1)
                    st.success(f":blue[Direkomendasikan] untuk menjual emas. Prediksi harga besok: :blue[{tomorrow_currency_price}/gram].\nAnda akan untung sebesar :blue[{profit_currency}]")
                else:
                    time.sleep(1)
                    st.warning(f":red[Tidak direkomendasikan] untuk menjual emas. Prediksi harga besok: :red[{tomorrow_currency_price}/gram]")
            # else:
            #     st.error("Data harga pada tanggal tersebut kosong. Tolong pilih tanggal lain yang tidak kosong.")

        if st.button("Tampilkan Rekomendasi"):
            tampilkan_rekomendasi()
        # tombol_rekomendasi()

    # REKOMENDASI BELI
    st.header("Rekomendasi Beli Emas")

    # Load data
    df = pd.read_csv('data/harga_emas_new2.csv')

    # Convert 'Tanggal' to datetime and set as index
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df.set_index('Tanggal', inplace=True)

    # Convert 'Price1' to float after cleaning
    df['Harga'] = df['Price1'].astype(str).str.replace('.', '').astype(float)

    # Drop unnecessary columns
    df.drop(['Price1', 'Price2', 'Price3', 'Price5', 'Price10', 'Price25', 'Price50', 'Price100'], axis=1, inplace=True)

    # Filter data untuk tahun 2021 - 2024
    df = df[(df.index.year >= pd.Timestamp.now().year - 3) & (df.index.year <= pd.Timestamp.now().year - 1)]

    # Extract month and year
    df['Month'] = df.index.month
    df['Year'] = df.index.year

    # Calculate average price for each month of each year
    monthly_avg_prices = df.groupby(['Year', 'Month'])['Harga'].mean().reset_index()

    # Identify the month with the lowest average price for each year
    recommended_months_per_year = monthly_avg_prices.loc[monthly_avg_prices.groupby('Year')['Harga'].idxmin()]
    recommended_month = monthly_avg_prices.loc[monthly_avg_prices.groupby('Month')['Harga'].idxmin()]
    
    # Data pivot untuk tabel
    monthly_avg_prices_pivot = monthly_avg_prices.pivot(index='Year', columns='Month', values='Harga')

    # Mapping month numbers to names
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
    
    # Tabel rata-rata harga per bulan
    st.write("Tabel rata-rata harga emas bulanan")
    
    # Mengubah format penulisan harga
    locale.setlocale(locale.LC_ALL, 'id_ID.UTF-8')
    def format_currency(value):
        return locale.format_string("%.2f", value, grouping=True)
    
    # Agar kolom bulan menggunakan nama bulan
    monthly_avg_prices_pivot.columns = [bulan[i] for i in monthly_avg_prices_pivot.columns]
    # Format 2 angka setelah desimal
    monthly_avg_prices_pivot = monthly_avg_prices_pivot.map(format_currency)
    # Format penulisan tahun, tanpa koma
    monthly_avg_prices_pivot.index = monthly_avg_prices_pivot.index.astype(str)
    # Tampilkan data harga sejak 3 tahun lalu
    # monthly_avg_prices_pivot = monthly_avg_prices_pivot(monthly_avg_prices_pivot['Year'].isin([current_year, current_year-1, current_year-2, current_year-3]))
    
    
    # format_tahun = str(monthly_avg_prices_pivot.index).replace(',', '')
    # monthly_avg_prices_pivot.index = format_tahun
    # monthly_avg_prices_pivot = monthly_avg_prices_pivot.map(lambda x: f"{x:.2f}")
    # monthly_avg_prices_pivot = locale.currency(monthly_avg_prices_pivot.map(lambda x: f"{x:.2f}"))
    # monthly_avg_prices_pivot = locale.currency(monthly_avg_prices_pivot)
    
    # df['Year'] = df['Year'].str.replace(',', '')
    st.dataframe(monthly_avg_prices_pivot)
    
    bulan_terendah = []
    frekuensi_bulan = {}

    # Display overall recommendation
    st.write("Rekomendasi beli:")
    for index, row in recommended_months_per_year.iterrows():
        year = row['Year']
        month = row['Month']
        month_name_id = bulan[month]
        
        bulan_terendah.append(month_name_id)
        # st.write(f"Bulan terendah : {', '.join(bulan_terendah)}.")
        st.success(f"Bulan terbaik untuk membeli emas pada tahun :blue[{year:.0f}] adalah bulan :red[{month_name_id}] berdasarkan tren harga historis.")

    for month in bulan_terendah:
        if month in frekuensi_bulan:
            frekuensi_bulan[month] += 1
        else:
            frekuensi_bulan[month] = 1
    
    # st.write(f"Frekuensi bulan: {frekuensi_bulan}")
    # st.write(f"Bulan terbanyak: {max(frekuensi_bulan, key=frekuensi_bulan.get)}")
    
    # Mengecek frekuensi bulan terpilih
    unique_frequences = set(frekuensi_bulan.values())
    
    if len(unique_frequences) == 1:
        # Jika frekuensi semua bulan yg terpilih sama
        st.caption(f"Pada 3 tahun terakhir, harga emas cenderung turun pada bulan :red[{', '.join(frekuensi_bulan.keys())}].")
    else:
        # Jika terdapat 1 bulan dengan frekuensi tertinggi
        bulan_terendah_terbesar = [bulan for bulan, freq in frekuensi_bulan.items() if freq == max(frekuensi_bulan.values())]
        st.caption(f"Berdasarkan data pada 3 tahun terakhir, harga emas cenderung turun pada bulan {', '.join(bulan_terendah_terbesar)}")