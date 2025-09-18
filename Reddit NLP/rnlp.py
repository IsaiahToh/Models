# ------------------------------
# 1. Imports
# ------------------------------
import pandas as pd
import numpy as np
import re
import pickle
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ------------------------------
# 2. Load Data
# ------------------------------
# Format: CSV with "text", "label" (0=neg, 1=neutral, 2=pos)
df = pd.read_csv("labeled_posts.csv")
texts = df["text"].astype(str).values
labels = df["label"].values

# ------------------------------
# 3. Preprocessing
# ------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)          # remove links
    text = re.sub(r"[^a-z\s]", "", text)         # remove punctuation/numbers
    text = " ".join([w for w in text.split() if w not in stop_words])
    return text

cleaned_texts = [clean_text(t) for t in texts]

# ------------------------------
# 4. Tokenization + Padding
# ------------------------------
vocab_size = 15000
max_length = 120
embedding_dim = 128

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(cleaned_texts)
sequences = tokenizer.texts_to_sequences(cleaned_texts)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")

X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42, stratify=labels
)

# ------------------------------
# 5. Sentiment Model (Deep Learning, from scratch)
# ------------------------------
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    LSTM(256, return_sequences=True),
    Dropout(0.5),
    LSTM(128),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(3, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              metrics=["accuracy"])

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=10,
    batch_size=64,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)]
)

loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)

y_pred = model.predict(X_test).argmax(axis=1)
print(classification_report(y_test, y_pred, digits=3))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ------------------------------
# 6. Topic Modeling (NMF on TF-IDF)
# ------------------------------
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
X_tfidf = vectorizer.fit_transform(cleaned_texts)

n_topics = 5
nmf = NMF(n_components=n_topics, random_state=42)
W = nmf.fit_transform(X_tfidf)
H = nmf.components_

feature_names = vectorizer.get_feature_names_out()

print("\nTop Topics:")
for i, topic in enumerate(H):
    top_words = [feature_names[j] for j in topic.argsort()[-10:]]
    print(f"Topic {i+1}: {', '.join(top_words)}")

# ------------------------------
# 7. Trending Phrases (n-grams + TF-IDF)
# ------------------------------
def get_trending_phrases(docs, top_n=15):
    vec = CountVectorizer(ngram_range=(2,3), stop_words="english", max_features=2000)
    X = vec.fit_transform(docs)
    freqs = np.asarray(X.sum(axis=0)).ravel()
    phrases = vec.get_feature_names_out()
    sorted_idx = freqs.argsort()[::-1]
    return [(phrases[i], freqs[i]) for i in sorted_idx[:top_n]]

trending_phrases = get_trending_phrases(cleaned_texts, top_n=15)

print("\nTrending Phrases:")
for phrase, freq in trending_phrases:
    print(f"{phrase}: {freq}")

# ------------------------------
# 8. Save Models + Tokenizer
# ------------------------------
model.save("sentiment_model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("nmf_model.pkl", "wb") as f:
    pickle.dump(nmf, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
