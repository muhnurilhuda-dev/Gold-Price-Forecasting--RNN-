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
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


st.set_page_config(page_title="nGold", page_icon=":bar_chart")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {vidibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# #user authenticarion
# names = ["admin"]
# usernames = ["admin"]

# #load password
# file_path = Path(__file__).parent / "hashed_pw.pkl"
# with file_path.open("rb") as file:
#     hashed_passwords = pickle.load(file)

# authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "ngold", "abcdef", cookie_expiry_days=30)

# names, authentication_status, usernames = authenticator.login("Login", "main")

# if authentication_status == False:
#     st.error("Username/Password is incorect")

# if authentication_status == None:
#     st.warning("Please Enter your username and password")
    
# if authentication_status:       

# Pilihan menu
st.sidebar.title(f"Welcome Nuril")
menu = ["Buyback", "Emas 1gr", "Emas 5gr", "Emas 10gr", "Emas 25gr", "Emas 100gr"]
choice = st.sidebar.selectbox("Menu", menu)
# authenticator.logout("Logout", "sidebar")

with open('style_css.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

if choice == "Buyback":
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
        # print(end_date)
            
        if start_date <= end_date:
            # Scraping data dan menyimpannya ke dalam file CSV
            data_list = scrape_data(start_date, end_date)
            save_to_csv(data_list, file_name)
    # Load data
    df = pd.read_csv('data/buy_1gr.csv')
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df['Tanggal'] = df['Tanggal'].dt.date
    df.set_index('Tanggal', inplace=True)
    df['Harga'] = (df['Pricebuy'])
    #load data lagi
    df2 = pd.read_csv('data/buy_1gr.csv')
    df2['Tanggal'] = pd.to_datetime(df2['Tanggal'])
    df2.set_index('Tanggal', inplace=True)
    # Calculate Weighted Moving Average
    weights = [3, 2, 1]
    weights = np.array(weights)
    wmavg = df['Pricebuy'].rolling(window=3, win_type=None).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    wmavg2 = df2['Pricebuy'].rolling(window=3, win_type=None).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    # tambah kolom "Keterangan"
    df['Keterangan'] = np.where(df['Pricebuy'].diff() > 0, 'Naik', 'Turun')
    # Add the latest prediction to the actual data for the next day
    latest_prediction = wmavg2.iloc[-1]
    next_day = wmavg2.index[-1] + pd.DateOffset(days=1)
    df2.loc[next_day, 'Pricebuy'] = latest_prediction
    # Load data
    df3 = pd.read_csv('data/emas_buypack.csv')
    df3['Tanggal'] = pd.to_datetime(df3['Tanggal'])
    df3.set_index('Tanggal', inplace=True)
    # WMA
    weights1 = [3, 2, 1]
    weights1 = np.array(weights1)
    wmavg1 = df3['Pricebuy'].rolling(window=3, win_type=None).apply(lambda x: (x * weights1).sum() / weights1.sum(), raw=True)
    
    # def calculate_accuracy_metrics(actual, predicted):
    #     mse = mean_squared_error(actual, predicted)
    #     mape = mean_absolute_percentage_error(actual, predicted)
    #     st.write(f"Mean Squared Error (MSE): {mse}")
    #     st.write(f"Mean Absolute Percentage Error (MAPE): {mape * 100:.2f}%")
    
    # # Check if 'Price10' column exists in df2
    # if 'Price10' in df2.columns and 'Price10' in wmavg2.columns:
    #     # Align the series by their indexes to ensure comparisons are made correctly
    #     common_index = wmavg2.index.intersection(df2.index)
    #     actual_values = df2.loc[common_index, 'Price10']
    #     predicted_values = wmavg2.loc[common_index]
    
    #     # Calculate accuracy metrics
    #     calculate_accuracy_metrics(actual_values, predicted_values)
    # else:
    #     print("Error: 'Price10' column not found in df2 or wmavg2. Please check the data.")

    def load_data():
        file = pd.read_csv('data/emas_buypack.csv')
        return file
    file = load_data()
    month_names = {
        1: 'Jan',
        2: 'Feb',
        3: 'Mar',
        4: 'Apr',
        5: 'Mei',
        6: 'Jun',
        7: 'Jul',
        8: 'Agus',
        9: 'Sept',
        10: 'Okt',
        11: 'Nov',
        12: 'Des'
    }
    # Mengubah kolom 'Date' menjadi tipe data datetime
    file['Tanggal'] = pd.to_datetime(file['Tanggal'])
    # Mendapatkan tahun dan bulan dari kolom 'Date'
    file['Year'] = file['Tanggal'].dt.year
    file['Month'] = file['Tanggal'].dt.month
    # Mengelompokkan data berdasarkan tahun dan bulan
    grouped_data = file.groupby(['Year', 'Month']).mean().reset_index()
    # Menambahkan kolom 'Trend' yang menunjukkan apakah harga emas naik atau turun
    grouped_data['Trend'] = grouped_data['Pricebuy'].diff().fillna(0)
    grouped_data['Trend'] = grouped_data['Trend'].apply(lambda x: 'Naik' if x > 0 else 'Turun' if x < 0 else 'Tidak Berubah')
    # grafik
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Pricebuy'], mode='lines', name='Harga Emas Aktual'))
    fig.add_trace(go.Scatter(x=df.index, y=wmavg, mode='lines', name='Prediksi Harga Emas'))
    # layout grafik
    fig.update_layout(
        xaxis_title='Harga Buypack 1 gr',
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
    
    def calculate_accuracy_metrics(df_predictions):
        actual_values = df_predictions['actual']
        predicted_values = df_predictions['predicted']

        mse = mean_squared_error(actual_values, predicted_values)
        mape = mean_absolute_percentage_error(actual_values, predicted_values)

        return mse, mape

    # Example DataFrame creation (replace with your actual data)
    df_predictions = pd.DataFrame({
        'actual': [100, 150, 200, 250, 300],
        'predicted': [110, 145, 195, 260, 290]
    })

    # Calculate MSE and MAPE
    mse, mape = calculate_accuracy_metrics(df_predictions)

    # Display the results
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2%}")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Harga Buyback Terakhir')
            st.write(f'Rp {df.round(0).tail(1)["Pricebuy"].values[0]:,.0f}')
            data_akhir = last_price = df.round(0).tail(1)["Pricebuy"].values[0]
            data_sebelum  = df.round(0).tail(2)["Pricebuy"].values[0]
            selisih = data_akhir - data_sebelum
            #logic emoji
            if data_akhir > data_sebelum:
                color = 'green'
                emoji = '▲'
            elif data_akhir < data_sebelum:
                color = 'red'
                emoji = '▼'
            else:
                color = 'yellow'
                emoji = '−'
            st.markdown(f'<span style="color:{color}">{emoji} Rp {selisih:,.0f}</span>', unsafe_allow_html=True)
        with col2:
            st.subheader('Prediksi Besok ')
            harini = datetime.now().date()
            st.write(f'Rp {df2.round(0).tail(1)["Pricebuy"].values[0]:,.0f}')
            dat_akhir = last_price1 = df2.round(0).tail(1)["Pricebuy"].values[0]
            dat_sebelum  = df.round(0).tail(1)["Pricebuy"].values[0]
            selisih1 = dat_akhir - dat_sebelum
            #logic emoji
            if dat_akhir > dat_sebelum:
                color = 'green'
                emoji = '▲'
            elif dat_akhir < dat_sebelum:
                color = 'red'
                emoji = '▼'
            else:
                color = 'yellow'
                emoji = '−'
            st.markdown(f'<span style="color:{color}">{emoji} Rp {selisih1:,.0f}</span>', unsafe_allow_html=True)
            naik = []
            turun = []
            for month in range(1, 13):
                monthly_data = grouped_data[grouped_data['Month'] == month]
                trend = monthly_data['Trend'].iloc[-1]
                month_name = month_names[month]
                if trend == 'Naik':
                    naik.append(month_name)
                elif trend == 'Turun':
                    turun.append(month_name)
            if naik:
                st.write("Periode naik:")
                st.write(", ".join(naik))
            else:
                st.write("Tidak ada periode naik")
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Harga Buyback 1 Gr')
            waktu = datetime.now().time()
            st.write(df.iloc[:, [1, 2]].round(0).tail(7))
        
            
        with col2:
            st.subheader('Simulasi Buyback')
            # Membaca data dari file CSV
            df4 = pd.read_csv('data/harga_emas.csv')
            # Mengubah kolom tanggal menjadi tipe data datetime
            df4['Tanggal'] = pd.to_datetime(df4['Tanggal'])
            # Menampilkan inputan tanggal
            gram = st.number_input("Satuan gram", 0)
            inputan = st.date_input("Tanggal Beli Emas")
            # Filter data berdasarkan tanggal inputan yang sama dengan tanggal di file CSV
            data_harga_inputan = df4[df4['Tanggal'] == pd.to_datetime(inputan)]
            # Memulai perkalian antara harga dan gram ketika tombol "Hitung" ditekan
            hitung_buy = st.button("Hitung")
            now = datetime.now().date()
            if inputan > now:
                st.write("Data harga emas belum tersedia.")
            elif hitung_buy and not data_harga_inputan.empty:
                harga = 0
                if gram in [100, 50, 25, 5, 3, 2]:
                    if gram == 100:
                        kolom_harga = 'Price100'
                    elif gram == 25:
                        kolom_harga = 'Price25'
                    elif gram == 10:
                        kolom_harga = 'Price10'
                    elif gram == 5:
                        kolom_harga = 'Price5'   
                    elif gram == 3:
                        kolom_harga = 'Price3'
                    elif gram == 2:
                        kolom_harga = 'Price2' 
                    harga = data_harga_inputan[kolom_harga].values[0]
                    harga_terakhir_gram = df['Harga'].iloc[-1]
                    total_harga2 = gram * harga_terakhir_gram
                    total2 = gram * harga
                    selisih = total_harga2 - harga
                    st.write("Harga Beli Emas {} gr Sesuai Tanggal Beli".format(gram))
                    st.write("<span style='color:black'>Rp. {:,.0f}</span>".format(abs(harga)), unsafe_allow_html=True)
                    st.write("Perkiraan Harga Emas {} gr:".format(gram))
                    st.write("<span style='color:black'>Rp. {:,.0f}</span>".format(abs(total_harga2)), unsafe_allow_html=True)
                    selisih = total_harga2 - harga
                    if selisih < 0:
                        st.write("Kerugian sebesar <span style='color:red'>Rp. {:,.0f}</span>".format(abs(selisih)), unsafe_allow_html=True)
                        st.text("Disarankan untuk jangan jual")
                    elif selisih > 0:
                        st.write("Keuntungan sebesar <span style='color:green'>Rp. {:,.0f}</span>".format(abs(selisih)), unsafe_allow_html=True)
                        st.write("Disarankan untuk Jual")
                    else:
                        st.write("Tidak untung tidak rugi")
                else:
                    kolom_harga = 'Price1'
                    harga = data_harga_inputan[kolom_harga].values[0]
                    harga_terakhir_gram = df['Harga'].iloc[-1]
                    total_harga2 = gram * harga_terakhir_gram
                    total2 = gram * harga
                    st.write("Harga Beli Emas {} gr Sesuai Tanggal Beli".format(gram))
                    st.write("<span style='color:black'>Rp. {:,.0f}</span>".format(abs(total2)), unsafe_allow_html=True)
                    st.write("Perkiraan Harga Emas {} gr:".format(gram))
                    st.write("<span style='color:black'>Rp. {:,.0f}</span>".format(abs(total_harga2)), unsafe_allow_html=True)
                    selisih = total_harga2 - total2
                    if selisih < 0:
                        st.write("Kerugian sebesar <span style='color:red'>Rp. {:,.0f}</span>".format(abs(selisih)), unsafe_allow_html=True)
                        st.text("Disarankan untuk jangan jual")
                    elif selisih > 0:
                        st.write("Keuntungan sebesar <span style='color:green'>Rp. {:,.0f}</span>".format(abs(selisih)), unsafe_allow_html=True)
                        st.write("Disarankan untuk Jual")
                    else:
                        st.write("Tidak untung tidak rugi")
            elif hitung_buy and data_harga_inputan.empty:
                st.write("Tidak ada data harga pada tanggal ini.")
    if waktu.hour > 9:
        def schedule_scraping():
            # Schedule fungsi scraping untuk dijalankan setiap hari pukul 09:00
            schedule.every().day.at("09:00:59").do(scrape_dan_simpan)
            # Tetap jalankan program
            while True:
                schedule.run_pending()
                time.sleep(1)
        if __name__ == '__main__':
            scrape_dan_simpan()
            schedule_scraping() 
elif choice == "Emas 1gr":  
    # Load data
    df = pd.read_csv('data/harga_emas.csv')
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df['Tanggal'] = df['Tanggal'].dt.date
    df.set_index('Tanggal', inplace=True)
        
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
        return data_list
    def save_to_csv(data_list, file_name):
        with open(file_name, 'a', newline='') as file:
            csv_writer = writer(file)
            csv_writer.writerows(data_list)
    def scrape_and_save():
        # Baca tanggal terakhir dari file CSV
        file_name = 'data/harga_emas.csv'
        df5 = pd.read_csv(file_name)
        last_date_str = df5.iloc[-1]['Tanggal']
        last_date = datetime.strptime(last_date_str, '%Y-%m-%d').date()
        
        start_date = last_date + timedelta(days=1)
        end_date = datetime.now().date()
        
        if start_date <= end_date:
            # Scraping data dan menyimpannya ke dalam file CSV
            data_list = scrape_data(start_date, end_date)
            save_to_csv(data_list, file_name)
    df1 = pd.read_csv('data/harga_emas.csv')
    df1['Tanggal'] = pd.to_datetime(df1['Tanggal'])
    df1['Tanggal'] = df1['Tanggal'].dt.date
    df1.set_index('Tanggal', inplace=True)
    df1['Harga'] = (df1['Price1'])
    #load data lagi
    df2 = pd.read_csv('data/harga_Emas.csv')
    df2['Tanggal'] = pd.to_datetime(df2['Tanggal'])
    df2.set_index('Tanggal', inplace=True)
    # WMA
    weights = [3, 2, 1]
    weights = np.array(weights)
    wmavg = df['Price1'].rolling(window=3, win_type=None).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    wmavg2 = df2['Price1'].rolling(window=3, win_type=None).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    df1['Keterangan'] = np.where(df1['Price1'].diff() > 0, 'Naik', 'Turun')
    latest_prediction = wmavg2.iloc[-1]
    next_day = wmavg2.index[-1] + pd.DateOffset(days=1)
    df2.loc[next_day, 'Price1'] = latest_prediction
    # Load data
    df3 = pd.read_csv('data/emas_th1gr.csv')
    df3['Tanggal'] = pd.to_datetime(df3['Tanggal'])
    df3.set_index('Tanggal', inplace=True)
    # WMA
    weights1 = [3, 2, 1]
    weights1 = np.array(weights1)
    wmavg1 = df3['Price1'].rolling(window=3, win_type=None).apply(lambda x: (x * weights1).sum() / weights1.sum(), raw=True)

    def load_data():
        file = pd.read_csv('data/emas_th1gr.csv')
        return file
    file = load_data()
    month_names = {
        1: 'Jan',
        2: 'Feb',
        3: 'Mar',
        4: 'Apr',
        5: 'Mei',
        6: 'Jun',
        7: 'Jul',
        8: 'Agus',
        9: 'Sept',
        10: 'Okt',
        11: 'Nov',
        12: 'Des'
    }
    # Mengubah kolom 'Date' menjadi tipe data datetime
    file['Tanggal'] = pd.to_datetime(file['Tanggal'])
    # Mendapatkan tahun dan bulan dari kolom 'Date'
    file['Year'] = file['Tanggal'].dt.year
    file['Month'] = file['Tanggal'].dt.month
    # Mengelompokkan data berdasarkan tahun dan bulan
    grouped_data = file.groupby(['Year', 'Month']).mean().reset_index()
    # Menambahkan kolom 'Trend' yang menunjukkan apakah harga emas naik atau turun
    grouped_data['Trend'] = grouped_data['Price1'].diff().fillna(0)
    grouped_data['Trend'] = grouped_data['Trend'].apply(lambda x: 'Naik' if x > 0 else 'Turun' if x < 0 else 'Tidak Berubah')
    # grafik
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Price1'], mode='lines', name='Harga Emas Aktual'))
    fig.add_trace(go.Scatter(x=df.index, y=wmavg, mode='lines', name='Prediksi Harga Emas'))
    # layout grafik
    fig.update_layout(
        xaxis_title='Grafik Harga Emas 1 Gr',
        yaxis_title='Harga Emas (Rupiah)',  
            xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label='1 w', step='day', stepmode='backward'),
                    dict(count=3, label='3 m', step='month', stepmode='backward'),
                    dict(count=6, label='6 m', step='month', stepmode='backward', visible=True),
                    dict(step='all')
                ])
            ),
            rangeslider=dict(visible=True),
            type='date'
        ),
        width=750,  # Lebar grafik (dalam piksel)
        height=400
    )    
    with st.container(): 
        st.plotly_chart(fig)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Harga Emas 1 Gr Terakhir')
            st.write(f'Rp {df.round(0).tail(1)["Price1"].values[0]:,.0f}')
            data_akhir = last_price = df.round(0).tail(1)["Price1"].values[0]
            data_sebelum  = df.round(0).tail(2)["Price1"].values[0]
            selisih = data_akhir - data_sebelum
            #logic emoji
            if data_akhir > data_sebelum:
                color = 'green'
                emoji = '▲'
            elif data_akhir < data_sebelum:
                color = 'red'
                emoji = '▼'
            else:
                color = 'yellow'
                emoji = '−'
            st.markdown(f'<span style="color:{color}">{emoji} Rp {selisih:,.0f}</span>', unsafe_allow_html=True)
        with col2:
            st.subheader('Prediksi Besok ')
            harini = datetime.now().date()
            st.write(f'Rp {df2.round(0).tail(1)["Price1"].values[0]:,.0f}')
            dat_akhir = last_price1 = df2.round(0).tail(1)["Price1"].values[0]
            dat_sebelum  = df.round(0).tail(1)["Price1"].values[0]
            selisih1 = dat_akhir - dat_sebelum
            #logic emoji
            if dat_akhir > dat_sebelum:
                color = 'green'
                emoji = '▲'
            elif dat_akhir < dat_sebelum:
                color = 'red'
                emoji = '▼'
            else:
                color = 'yellow'
                emoji = '−'
            st.markdown(f'<span style="color:{color}">{emoji} Rp {selisih1:,.0f}</span>', unsafe_allow_html=True)
            naik = []
            turun = []
            for month in range(1, 13):
                monthly_data = grouped_data[grouped_data['Month'] == month]
                trend = monthly_data['Trend'].iloc[-1]
                month_name = month_names[month]
                if trend == 'Naik':
                    naik.append(month_name)
                elif trend == 'Turun':
                    turun.append(month_name)
            if turun:
                st.write("Periode turun:")
                st.write(", ".join(turun))
            else:
                st.write("Tidak ada periode turun")
            
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Harga Emas 1 Gr')
            waktu = datetime.now().time()
            st.write(df1.iloc[:, [8, 9]].round(0).tail(7))
            
        with col2:
            st.subheader('Konversi Harga Emas')
            harga_terakhir_gram = df['Price1'].iloc[-1]
            konversi = st.number_input("Dalam Satuan gram", 0)
            hitung = st.button ("hitung")
            if hitung :
                hasil = konversi * harga_terakhir_gram
                st.write ("Rp {:,.0f}".format(hasil))
                st.write ("Disarankan beli sesuai periode turun")
                
    if waktu.hour > 9:
        def schedule_scraping():
            # Schedule fungsi scraping untuk dijalankan setiap hari pukul 09:00
            schedule.every().day.at("09:00:01").do(scrape_and_save)
            # Tetap jalankan program
            while True:
                schedule.run_pending()
                time.sleep(1)
        if __name__ == '__main__':
            scrape_and_save()
            schedule_scraping() 
               
elif choice == "Emas 5gr":
    # Load data
    df = pd.read_csv('data/harga_emas.csv')
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df['Tanggal'] = df['Tanggal'].dt.date
    df.set_index('Tanggal', inplace=True)
    df1 = pd.read_csv('data/harga_emas.csv')
    df1['Tanggal'] = pd.to_datetime(df1['Tanggal'])
    df1['Tanggal'] = df1['Tanggal'].dt.date
    df1.set_index('Tanggal', inplace=True)
    df1['Harga'] = (df1['Price5'])
    #load data lagi
    df2 = pd.read_csv('data/harga_emas.csv')
    df2['Tanggal'] = pd.to_datetime(df2['Tanggal'])
    df2.set_index('Tanggal', inplace=True)
    # WMA
    weights = [3, 2, 1]
    weights = np.array(weights)
    wmavg = df['Price5'].rolling(window=3, win_type=None).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    wmavg2 = df2['Price5'].rolling(window=3, win_type=None).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    df1['Keterangan'] = np.where(df1['Price5'].diff() > 0, 'Naik', 'Turun')
    latest_prediction = wmavg2.iloc[-1]
    next_day = wmavg2.index[-1] + pd.DateOffset(days=1)
    df2.loc[next_day, 'Price5'] = latest_prediction
    # Load data
    df3 = pd.read_csv('data/emas_th5gr.csv')
    df3['Tanggal'] = pd.to_datetime(df3['Tanggal'])
    df3.set_index('Tanggal', inplace=True)
    # WMA
    weights1 = [3, 2, 1]
    weights1 = np.array(weights1)
    wmavg1 = df3['Harga'].rolling(window=3, win_type=None).apply(lambda x: (x * weights1).sum() / weights1.sum(), raw=True)

    def load_data():
        file = pd.read_csv('data/emas_th5gr.csv')
        return file
    file = load_data()
    month_names = {
        1: 'Jan',
        2: 'Feb',
        3: 'Mar',
        4: 'Apr',
        5: 'Mei',
        6: 'Jun',
        7: 'Jul',
        8: 'Agus',
        9: 'Sept',
        10: 'Okt',
        11: 'Nov',
        12: 'Des'
    }
    # Mengubah kolom 'Date' menjadi tipe data datetime
    file['Tanggal'] = pd.to_datetime(file['Tanggal'])
    # Mendapatkan tahun dan bulan dari kolom 'Date'
    file['Year'] = file['Tanggal'].dt.year
    file['Month'] = file['Tanggal'].dt.month
    # Mengelompokkan data berdasarkan tahun dan bulan
    grouped_data = file.groupby(['Year', 'Month']).mean().reset_index()
    # Menambahkan kolom 'Trend' yang menunjukkan apakah harga emas naik atau turun
    grouped_data['Trend'] = grouped_data['Harga'].diff().fillna(0)
    grouped_data['Trend'] = grouped_data['Trend'].apply(lambda x: 'Naik' if x > 0 else 'Turun' if x < 0 else 'Tidak Berubah')
    # grafik
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Price5'], mode='lines', name='Harga Emas Aktual'))
    fig.add_trace(go.Scatter(x=df.index, y=wmavg, mode='lines', name='Prediksi Harga Emas'))
    # layout
    fig.update_layout(
        xaxis_title='Grafik Harga Emas 5 Gr',
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
    with st.container(): 
        st.plotly_chart(fig)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Harga Emas 5 Gr Terakhir')
            st.write(f'Rp {df.round(0).tail(1)["Price5"].values[0]:,.0f}')
            data_akhir = last_price = df.round(0).tail(1)["Price5"].values[0]
            data_sebelum  = df.round(0).tail(2)["Price5"].values[0]
            selisih = data_akhir - data_sebelum
            #logic emoji
            if data_akhir > data_sebelum:
                color = 'green'
                emoji = '▲'
            elif data_akhir < data_sebelum:
                color = 'red'
                emoji = '▼'
            else:
                color = 'yellow'
                emoji = '−'
            st.markdown(f'<span style="color:{color}">{emoji} Rp {selisih:,.0f}</span>', unsafe_allow_html=True)
        with col2:
            st.subheader('Prediksi Besok ')
            harini = datetime.now().date()
            st.write(f'Rp {df2.round(0).tail(1)["Price5"].values[0]:,.0f}')
            dat_akhir = last_price1 = df2.round(0).tail(1)["Price5"].values[0]
            dat_sebelum  = df.round(0).tail(1)["Price5"].values[0]
            selisih1 = dat_akhir - dat_sebelum
            #logic emoji
            if dat_akhir > dat_sebelum:
                color = 'green'
                emoji = '▲'
            elif dat_akhir < dat_sebelum:
                color = 'red'
                emoji = '▼'
            else:
                color = 'yellow'
                emoji = '−'
            st.markdown(f'<span style="color:{color}">{emoji} Rp {selisih1:,.0f}</span>', unsafe_allow_html=True)
            naik = []
            turun = []
            for month in range(1, 13):
                monthly_data = grouped_data[grouped_data['Month'] == month]
                trend = monthly_data['Trend'].iloc[-1]
                month_name = month_names[month]
                if trend == 'Naik':
                    naik.append(month_name)
                elif trend == 'Turun':
                    turun.append(month_name)
            if turun:
                st.write("Periode turun:")
                st.write(", ".join(turun))
            else:
                st.write("Tidak ada periode turun")
        
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Harga Emas 5 Gr')   
            st.write(df1.iloc[:, [8, 9]].round(0).tail(7))
            
        with col2:
            st.subheader('Konversi Harga Emas')
            harga_terakhir_gram = df['Price1'].iloc[-1]
            konversi = st.number_input("Dalam Satuan gram", 0)
            hitung = st.button ("hitung")
            if hitung :
                hasil = konversi * harga_terakhir_gram
                st.write ("Rp {:,.0f}".format(hasil))
                st.write ("Disarankan beli sesuai periode turun")
elif choice == "Emas 10gr":
    # Load data
    df = pd.read_csv('data/harga_emas.csv')
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df['Tanggal'] = df['Tanggal'].dt.date
    df.set_index('Tanggal', inplace=True)
    df1 = pd.read_csv('data/harga_emas.csv')
    df1['Tanggal'] = pd.to_datetime(df1['Tanggal'])
    df1['Tanggal'] = df1['Tanggal'].dt.date
    df1.set_index('Tanggal', inplace=True)
    df1['Harga'] = (df1['Price10'])
    #load data lagi
    df2 = pd.read_csv('data/harga_emas.csv')
    df2['Tanggal'] = pd.to_datetime(df2['Tanggal'])
    df2.set_index('Tanggal', inplace=True)
    # WMA
    weights = [3, 2, 1]
    weights = np.array(weights)
    wmavg = df['Price10'].rolling(window=3, win_type=None).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    wmavg2 = df2['Price10'].rolling(window=3, win_type=None).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    df1['Keterangan'] = np.where(df1['Price10'].diff() > 0, 'Naik', 'Turun')
    latest_prediction = wmavg2.iloc[-1]
    next_day = wmavg2.index[-1] + pd.DateOffset(days=1)
    df2.loc[next_day, 'Price10'] = latest_prediction
    # Load data
    df3 = pd.read_csv('data/emas_th10gr.csv')
    df3['Tanggal'] = pd.to_datetime(df3['Tanggal'])
    df3.set_index('Tanggal', inplace=True)
    # WMA
    weights1 = [3, 2, 1]
    weights1 = np.array(weights1)
    wmavg1 = df3['Harga'].rolling(window=3, win_type=None).apply(lambda x: (x * weights1).sum() / weights1.sum(), raw=True)

    def load_data():
        file = pd.read_csv('data/emas_th10gr.csv')
        return file
    file = load_data()
    month_names = {
        1: 'Jan',
        2: 'Feb',
        3: 'Mar',
        4: 'Apr',
        5: 'Mei',
        6: 'Jun',
        7: 'Jul',
        8: 'Agus',
        9: 'Sept',
        10: 'Okt',
        11: 'Nov',
        12: 'Des'
    }
    # Mengubah kolom 'Date' menjadi tipe data datetime
    file['Tanggal'] = pd.to_datetime(file['Tanggal'])
    # Mendapatkan tahun dan bulan dari kolom 'Date'
    file['Year'] = file['Tanggal'].dt.year
    file['Month'] = file['Tanggal'].dt.month
    # Mengelompokkan data berdasarkan tahun dan bulan
    grouped_data = file.groupby(['Year', 'Month']).mean().reset_index()
    # Menambahkan kolom 'Trend' yang menunjukkan apakah harga emas naik atau turun
    grouped_data['Trend'] = grouped_data['Harga'].diff().fillna(0)
    grouped_data['Trend'] = grouped_data['Trend'].apply(lambda x: 'Naik' if x > 0 else 'Turun' if x < 0 else 'Tidak Berubah')
    # grafik
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Price10'], mode='lines', name='Harga Emas Aktual'))
    fig.add_trace(go.Scatter(x=df.index, y=wmavg, mode='lines', name='Prediksi Harga Emas'))
    # layout grafik
    fig.update_layout(
        xaxis_title='Grafik Harga Emas 10 Gr',
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
        
    with st.container(): 
        st.plotly_chart(fig)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Harga Emas 10 Gr Terakhir')
            st.write(f'Rp {df.round(0).tail(1)["Price10"].values[0]:,.0f}')
            data_akhir = last_price = df.round(0).tail(1)["Price10"].values[0]
            data_sebelum  = df.round(0).tail(2)["Price10"].values[0]
            selisih = data_akhir - data_sebelum
            #logic emoji
            if data_akhir > data_sebelum:
                color = 'green'
                emoji = '▲'
            elif data_akhir < data_sebelum:
                color = 'red'
                emoji = '▼'
            else:
                color = 'yellow'
                emoji = '−'
            st.markdown(f'<span style="color:{color}">{emoji} Rp {selisih:,.0f}</span>', unsafe_allow_html=True)
        with col2:
            st.subheader('Prediksi Besok ')
            harini = datetime.now().date()
            st.write(f'Rp {df2.round(0).tail(1)["Price10"].values[0]:,.0f}')
            dat_akhir = last_price1 = df2.round(0).tail(1)["Price10"].values[0]
            dat_sebelum  = df.round(0).tail(1)["Price10"].values[0]
            selisih1 = dat_akhir - dat_sebelum
            #logic emoji
            if dat_akhir > dat_sebelum:
                color = 'green'
                emoji = '▲'
            elif dat_akhir < dat_sebelum:
                color = 'red'
                emoji = '▼'
            else:
                color = 'yellow'
                emoji = '−'
            st.markdown(f'<span style="color:{color}">{emoji} Rp {selisih1:,.0f}</span>', unsafe_allow_html=True)
            naik = []
            turun = []
            for month in range(1, 13):
                monthly_data = grouped_data[grouped_data['Month'] == month]
                trend = monthly_data['Trend'].iloc[-1]
                month_name = month_names[month]
                if trend == 'Naik':
                    naik.append(month_name)
                elif trend == 'Turun':
                    turun.append(month_name)
            if turun:
                st.write("Periode turun:")
                st.write(", ".join(turun))
            else:
                st.write("Tidak ada periode turun")
        
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Harga Emas 10 Gr')    
            st.write(df1.iloc[:, [8, 9]].round(0).tail(7))
            
        with col2:
            st.subheader('Konversi Harga Emas')
            harga_terakhir_gram = df['Price1'].iloc[-1]
            konversi = st.number_input("Dalam Satuan gram", 0)
            hitung = st.button ("hitung")
            if hitung :
                hasil = konversi * harga_terakhir_gram
                st.write ("Rp {:,.0f}".format(hasil))
                st.write ("Disarankan beli sesuai periode turun")

elif choice == "Emas 25gr":
    # Load data
    df = pd.read_csv('data/harga_emas.csv')
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df['Tanggal'] = df['Tanggal'].dt.date
    df.set_index('Tanggal', inplace=True)
    df1 = pd.read_csv('data/harga_emas.csv')
    df1['Tanggal'] = pd.to_datetime(df1['Tanggal'])
    df1['Tanggal'] = df1['Tanggal'].dt.date
    df1.set_index('Tanggal', inplace=True)
    df1['Harga'] = (df1['Price25'])
    #load data lagi
    df2 = pd.read_csv('data/harga_emas.csv')
    df2['Tanggal'] = pd.to_datetime(df2['Tanggal'])
    df2.set_index('Tanggal', inplace=True)
    # WMA
    weights = [3, 2, 1]
    weights = np.array(weights)
    wmavg = df['Price25'].rolling(window=3, win_type=None).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    wmavg2 = df2['Price25'].rolling(window=3, win_type=None).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    df1['Keterangan'] = np.where(df1['Price25'].diff() > 0, 'Naik', 'Turun')
    latest_prediction = wmavg2.iloc[-1]
    next_day = wmavg2.index[-1] + pd.DateOffset(days=1)
    df2.loc[next_day, 'Price25'] = latest_prediction
    # Load data
    df3 = pd.read_csv('data/emas_th25gr.csv')
    df3['Tanggal'] = pd.to_datetime(df3['Tanggal'])
    df3.set_index('Tanggal', inplace=True)
    # WMA
    weights1 = [3, 2, 1]
    weights1 = np.array(weights1)
    wmavg1 = df3['Harga'].rolling(window=3, win_type=None).apply(lambda x: (x * weights1).sum() / weights1.sum(), raw=True)
    def load_data():
        file = pd.read_csv('data/emas_th25gr.csv')
        return file
    file = load_data()
    month_names = {
        1: 'Jan',
        2: 'Feb',
        3: 'Mar',
        4: 'Apr',
        5: 'Mei',
        6: 'Jun',
        7: 'Jul',
        8: 'Agus',
        9: 'Sept',
        10: 'Okt',
        11: 'Nov',
        12: 'Des'
    }
    # Mengubah kolom 'Date' menjadi tipe data datetime
    file['Tanggal'] = pd.to_datetime(file['Tanggal'])
    # Mendapatkan tahun dan bulan dari kolom 'Date'
    file['Year'] = file['Tanggal'].dt.year
    file['Month'] = file['Tanggal'].dt.month
    # Mengelompokkan data berdasarkan tahun dan bulan
    grouped_data = file.groupby(['Year', 'Month']).mean().reset_index()
    # Menambahkan kolom 'Trend' yang menunjukkan apakah harga emas naik atau turun
    grouped_data['Trend'] = grouped_data['Harga'].diff().fillna(0)
    grouped_data['Trend'] = grouped_data['Trend'].apply(lambda x: 'Naik' if x > 0 else 'Turun' if x < 0 else 'Tidak Berubah')
    # grafik
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Price25'], mode='lines', name='Harga Emas Aktual'))
    fig.add_trace(go.Scatter(x=df.index, y=wmavg, mode='lines', name='Prediksi Harga Emas'))
    # layout grafik
    fig.update_layout(
        xaxis_title='Grafik Harga Emas 25 Gr',
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
        
    with st.container(): 
        st.plotly_chart(fig)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Harga Emas 25 Gr Terakhir')
            st.write(f'Rp {df.round(0).tail(1)["Price25"].values[0]:,.0f}')
            data_akhir = last_price = df.round(0).tail(1)["Price25"].values[0]
            data_sebelum  = df.round(0).tail(2)["Price25"].values[0]
            selisih = data_akhir - data_sebelum
            #logic emoji
            if data_akhir > data_sebelum:
                color = 'green'
                emoji = '▲'
            elif data_akhir < data_sebelum:
                color = 'red'
                emoji = '▼'
            else:
                color = 'yellow'
                emoji = '−'
            st.markdown(f'<span style="color:{color}">{emoji} Rp {selisih:,.0f}</span>', unsafe_allow_html=True)
        with col2:
            st.subheader('Prediksi Besok ')
            harini = datetime.now().date()
            st.write(f'Rp {df2.round(0).tail(1)["Price25"].values[0]:,.0f}')
            dat_akhir = last_price1 = df2.round(0).tail(1)["Price25"].values[0]
            dat_sebelum  = df.round(0).tail(1)["Price25"].values[0]
            selisih1 = dat_akhir - dat_sebelum
            #logic emoji
            if dat_akhir > dat_sebelum:
                color = 'green'
                emoji = '▲'
            elif dat_akhir < dat_sebelum:
                color = 'red'
                emoji = '▼'
            else:
                color = 'yellow'
                emoji = '−'
            st.markdown(f'<span style="color:{color}">{emoji} Rp {selisih1:,.0f}</span>', unsafe_allow_html=True)
            naik = []
            turun = []
            for month in range(1, 13):
                monthly_data = grouped_data[grouped_data['Month'] == month]
                trend = monthly_data['Trend'].iloc[-1]
                month_name = month_names[month]
                if trend == 'Naik':
                    naik.append(month_name)
                elif trend == 'Turun':
                    turun.append(month_name)
            if turun:
                st.write("Periode turun:")
                st.write(", ".join(turun))
            else:
                st.write("Tidak ada periode turun")
            
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Harga Emas 25 Gr')    
            st.write(df1.iloc[:, [8, 9]].round(0).tail(7))
            
        with col2:
            st.subheader('Konversi Harga Emas')
            harga_terakhir_gram = df['Price1'].iloc[-1]
            konversi = st.number_input("Dalam Satuan gram", 0)
            hitung = st.button ("hitung")
            if hitung :
                hasil = konversi * harga_terakhir_gram
                st.write ("Rp {:,.0f}".format(hasil))
                st.write ("Disarankan beli sesuai periode turun")
                
elif choice == "Emas 100gr":
    # Load data
    df = pd.read_csv('data/harga_emas.csv')
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df['Tanggal'] = df['Tanggal'].dt.date
    df.set_index('Tanggal', inplace=True)
    df1 = pd.read_csv('data/harga_emas.csv')
    df1['Tanggal'] = pd.to_datetime(df1['Tanggal'])
    df1['Tanggal'] = df1['Tanggal'].dt.date
    df1.set_index('Tanggal', inplace=True)
    df1['Harga'] = (df1['Price100'])
    #load data lagi
    df2 = pd.read_csv('data/harga_emas.csv')
    df2['Tanggal'] = pd.to_datetime(df2['Tanggal'])
    df2.set_index('Tanggal', inplace=True)
    # WMA
    weights = [3, 2, 1]
    weights = np.array(weights)
    wmavg = df['Price100'].rolling(window=3, win_type=None).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    wmavg2 = df2['Price100'].rolling(window=3, win_type=None).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    df1['Keterangan'] = np.where(df1['Price100'].diff() > 0, 'Naik', 'Turun')
    latest_prediction = wmavg2.iloc[-1]
    next_day = wmavg2.index[-1] + pd.DateOffset(days=1)
    df2.loc[next_day, 'Price100'] = latest_prediction
    # Load data
    df3 = pd.read_csv('data/emas_th100gr.csv')
    df3['Tanggal'] = pd.to_datetime(df3['Tanggal'])
    df3.set_index('Tanggal', inplace=True)
    #new
    
    def load_data():
        file = pd.read_csv('data/emas_th100gr.csv')
        return file
    file = load_data()
    month_names = {
        1: 'Jan',
        2: 'Feb',
        3: 'Mar',
        4: 'Apr',
        5: 'Mei',
        6: 'Jun',
        7: 'Jul',
        8: 'Agus',
        9: 'Sept',
        10: 'Okt',
        11: 'Nov',
        12: 'Des'
    }
    # Mengubah kolom 'Date' menjadi tipe data datetime
    file['Tanggal'] = pd.to_datetime(file['Tanggal'])
    # Mendapatkan tahun dan bulan dari kolom 'Date'
    file['Year'] = file['Tanggal'].dt.year
    file['Month'] = file['Tanggal'].dt.month
    # Mengelompokkan data berdasarkan tahun dan bulan
    grouped_data = file.groupby(['Year', 'Month']).mean().reset_index()
    # Menambahkan kolom 'Trend' yang menunjukkan apakah harga emas naik atau turun
    grouped_data['Trend'] = grouped_data['Harga'].diff().fillna(0)
    grouped_data['Trend'] = grouped_data['Trend'].apply(lambda x: 'Naik' if x > 0 else 'Turun' if x < 0 else 'Tidak Berubah')
    # grafik
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Price100'], mode='lines', name='Harga Emas Aktual'))
    fig.add_trace(go.Scatter(x=df.index, y=wmavg, mode='lines', name='Prediksi Harga Emas'))
    # layout grafik
    fig.update_layout(
        xaxis_title='Grafik Harga Emas 100 Gr',
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
        height=400,
    )
        
    with st.container(): 
        st.plotly_chart(fig)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Harga Emas 100 Gr Terakhir')
            st.write(f'Rp {df.round(0).tail(1)["Price100"].values[0]:,.0f}')
            data_akhir = last_price = df.round(0).tail(1)["Price100"].values[0]
            data_sebelum  = df.round(0).tail(2)["Price100"].values[0]
            selisih = data_akhir - data_sebelum
            #logic emoji
            if data_akhir > data_sebelum:
                color = 'green'
                emoji = '▲'
            elif data_akhir < data_sebelum:
                color = 'red'
                emoji = '▼'
            else:
                color = 'yellow'
                emoji = '−'
            st.markdown(f'<span style="color:{color}">{emoji} Rp {selisih:,.0f}', unsafe_allow_html=True)
        with col2:
            st.subheader('Prediksi Besok ')
            harini = datetime.now().date()
            st.write(f'Rp {df2.round(0).tail(1)["Price100"].values[0]:,.0f}')
            dat_akhir = last_price1 = df2.round(0).tail(1)["Price100"].values[0]
            dat_sebelum  = df.round(0).tail(1)["Price100"].values[0]
            selisih1 = dat_akhir - dat_sebelum
            #logic emoji
            if dat_akhir > dat_sebelum:
                color = 'green'
                emoji = '▲'
            elif dat_akhir < dat_sebelum:
                color = 'red'
                emoji = '▼'
            else:
                color = 'yellow'
                emoji = '−'
            st.markdown(f'<span style="color:{color}">{emoji} Rp {selisih1:,.0f}</span>', unsafe_allow_html=True)
            naik = []
            turun = []
            for month in range(1, 13):
                monthly_data = grouped_data[grouped_data['Month'] == month]
                trend = monthly_data['Trend'].iloc[-1]
                month_name = month_names[month]
                if trend == 'Naik':
                    naik.append(month_name)
                elif trend == 'Turun':
                    turun.append(month_name)
            if turun:
                st.write("Periode turun:")
                st.write(", ".join(turun))
            else:
                st.write("Tidak ada periode turun")
        
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Harga Emas 100 Gr')    
            st.write(df1.iloc[:, [8, 9]].round(0).tail(7))
            
        with col2:
            st.subheader('Konversi Harga Emas')
            harga_terakhir_gram = df['Price1'].iloc[-1]
            konversi = st.number_input("Dalam Satuan gram", 0)
            hitung = st.button ("hitung")
            if hitung :
                hasil = konversi * harga_terakhir_gram
                st.write ("Rp {:,.0f}".format(hasil))
                st.write ("Disarankan beli sesuai periode turun")
st.write('*Jika pembaharuan data belum muncul pada pukul 09.00 maka lakukan reload pada browser Anda')
