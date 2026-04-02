import random
import streamlit as st
import joblib
import pandas as pd
import utils

@st.cache_resource
def load_model():
    return joblib.load("chatbot_model.pkl")

@st.cache_data
def load_data():
    df_cb = pd.read_csv("data/tattoo_chatbot_data.csv")
    rspns = df_cb.groupby("intent")["response"].agg(list).to_dict()
    return df_cb, rspns

with st.spinner("Loading tattoo chatbot..."):
    model = load_model()
    df, responses = load_data()

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

st.title("Sweeney Ink Tattoo Chatbot")
st.text("v1.2")

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Your question", key="user_input")
    submitted = st.form_submit_button("Ask")

if submitted and user_input:
    cleaned_input = utils.normalize_text(user_input)
    intent = model.predict([cleaned_input])[0]

    intent_responses = responses.get(intent, [])
    if intent_responses:
        response = random.choice(intent_responses)
    else:
        response = "I’m not sure how to answer that yet."

    st.write(f"**Question:** {user_input}")
    st.write(f"**Intent:** {intent}")
    st.write(f"**Response:** {response}")