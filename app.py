import joblib
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def normalize_text(text):
    text = text.lower().strip()
    # common
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

    # normalize repeated punctuation
    text = re.sub(r"[!?]+", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def chatbot_reply(user_message):
    cleaned = normalize_text(user_message)
    intent = model.predict([cleaned])[0]
    return responses[intent][0]

# Load data
df = pd.read_csv("data/tattoo_chatbot_data.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
X = df["user_input"].apply(normalize_text)
y = df["intent"]

intent_counts = y.value_counts()
print("Intent counts:\n")
print(intent_counts)

# split the data into 80-20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
    ngram_range=(1,2),
    lowercase=True,
    min_df=2,
    sublinear_tf=True
    )),
    ("clf", LogisticRegression(
        max_iter=1000)
    )
])

# Train
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print(classification_report(y_test, preds, zero_division=0))

# Simple response lookup
responses = df.groupby("intent")["response"].apply(list).to_dict()


# Test
while True:
    msg = input("You: ")
    if msg.lower() in ["quit", "exit"]:
        break
    print("Bot:", chatbot_reply(msg))