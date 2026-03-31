import random
import re
import streamlit as st
import joblib
import pandas as pd

def normalize_text(text):
    text = text.lower().strip()
    replacements = {
        r"\bu\b": "you",
        r"\bur\b": "your",
        r"\bappt\b": "appointment",
        r"\bavail\b": "availability",
        r"\bloc\b": "location",
        r"\baddr\b": "address",
        r"\binfo\b": "information",
        r"\bwa\b": "washington",
    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"[!?]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

model = joblib.load("chatbot_model.pkl")

df = pd.read_csv("data/tattoo_chatbot_data.csv")
responses = df.groupby("intent")["response"].apply(list).to_dict()

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

st.title("Sweeney Ink Tattoo Chatbot")
st.text("v1.0")

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Your question", key="user_input")
    submitted = st.form_submit_button("Ask")

if submitted and user_input:
    cleaned_input = normalize_text(user_input)
    intent = model.predict([cleaned_input])[0]

    intent_responses = responses.get(intent, [])
    if intent_responses:
        response = random.choice(intent_responses)
    else:
        response = "I’m not sure how to answer that yet."

    st.write(f"**Question:** {user_input}")
    st.write(f"**Intent:** {intent}")
    st.write(f"**Response:** {response}")