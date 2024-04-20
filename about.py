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

gambar = 'jpg/about.jpg'
st.image(gambar, use_column_width=True)



