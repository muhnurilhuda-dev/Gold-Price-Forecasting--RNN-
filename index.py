import streamlit as st
import subprocess

st.set_page_config(page_title="dashboard", layout="wide")

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {vidibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

button_style = """
<style>
.button:hover {
    background-color: blue !important;
}

.button:active {
    background-color: blue !important;
}
</style>
"""
st.markdown(button_style, unsafe_allow_html=True)

with st.container():
    col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12  = st.columns(12)
    with col10:
        home = st.button('Home')
    gambar = 'jpg/gk.jpg'
    st.image(gambar, use_column_width=True)
    with col11:
        about = st.button('about')
        if about:
            subprocess.run(["streamlit", "run", "about.py"])
    with col12:
        login = st.button('Login')
        if login:
            subprocess.run(["streamlit", "run", "rnn.py"])



