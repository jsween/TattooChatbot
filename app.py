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
def chatbot_reply(user_message):
    cleaned = utils.normalize_text(user_message)
    intnt = model.predict([cleaned])[0]

    intnt_responses = responses.get(intnt, [])

    if not intnt_responses:
        return "Sorry, I'm not sure how to answer that."

    if len(intnt_responses) == 1:
        return intnt_responses[0]

    # Use session_state so no-repeat logic persists across reruns
    previous = st.session_state.last_response.get(intnt)
    choices = [r for r in intnt_responses if r != previous]
    picked = random.choice(choices)
    st.session_state.last_response[intnt] = picked

    return picked

with st.spinner("Loading tattoo chatbot..."):
    model = load_model()
    df, responses = load_data()

if "last_response" not in st.session_state:
    st.session_state.last_response = {}

st.title("Sweeney Ink Tattoo Chatbot")
st.text("v1.4")

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Your question", key="user_input")
    submitted = st.form_submit_button("Ask")

if submitted and user_input:
    cleaned_input = utils.normalize_text(user_input)
    intent = model.predict([cleaned_input])[0]

    if submitted and user_input:
        intent = model.predict([utils.normalize_text(user_input)])[0]
        response = chatbot_reply(user_input)

        st.write(f"🤖 **Response:**")
        st.write(response)
        st.write()
        st.write(f"**Last Question Asked:** {user_input}")
        st.write(f"**Determined Intent:** {intent}")