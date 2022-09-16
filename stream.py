import streamlit as st
from summ import *


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()


with header:
    st.title("Text Summarization")
    st.text("I summarize text provided by user")

with dataset:
    txt = st.text_area("Input the text you want","Start typing")

    if st.button("Generate the text"):
        generated_text = generate_summary(txt)
        st.write(generated_text)
        st.snow()
