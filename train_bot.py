import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import utils
import random
import numpy as np
last_response = {}  # tracks last response per intent to avoid repetition

def chatbot_reply(user_message):
    # normalize the input
    cleaned = utils.normalize_text(user_message)
    # get confideneore
    proba = model.predict_proba([cleaned])[0]
    # if model isn't confident enough, fallback to generic response
    if proba.max() < 0.4:
        return random.choice(responses["fallback_unknown"])
    # get the indices of the top 3 most confident intents
    top3_indices = np.argsort(proba)[-3:]
    top3_intents = model.classes_[top3_indices]
    top3_probs = proba[top3_indices]
    # choose one of top 3 intents, weighted by confidence score
    chosen_intent = random.choices(top3_intents, weights=top3_probs, k=1)[0]

    # get all responses
    options = responses[chosen_intent]
    # if more than 1 response, filter out the last one used
    if len(options) > 1:
        last = last_response.get(chosen_intent)
        options = [r for r in options if r != last]
    # pick a random response to prevent repetition
    reply = random.choice(options)
    # store most recent response
    last_response[chosen_intent] = reply
    return reply


# Load data
df = pd.read_csv("data/tattoo_chatbot_data.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
X = df["user_input"].apply(utils.normalize_text)
y = df["intent"]

intent_counts = y.value_counts()
print("Intent counts:\n")
print(intent_counts)

# split the data 80-20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# builds a pipeline to normalize and classify in 1 call
model = Pipeline([
    # converts text into numbers
    ("tfidf", TfidfVectorizer(
    # analyze 1 and 2 word pairs
    ngram_range=(1,2),
    lowercase=True,
    # ignore words that appear in less than 2 examples
    min_df=2,
    # reduce effect of common words
    sublinear_tf=True
    )),
    # use logistic regression to classify intent
    ("clf", LogisticRegression(
        # give 1,000 iterations to converge
        max_iter=1000)
    )
])

# train
model.fit(X_train, y_train)

# evaluate
preds = model.predict(X_test)
print(classification_report(y_test, preds, zero_division=0))

# create a simple response lookup
responses = df.groupby("intent")["response"].apply(list).to_dict()
# save trained model
joblib.dump(model, "chatbot_model.pkl")

# manually test the chatbot
while True:
    msg = input("You: ")
    if msg.lower() in ["quit", "exit", "q"]:
        print("Goodbye")
        break
    print("Bot:", chatbot_reply(msg))