import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("reddit_labeled_data.csv")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text) # remove URLs
    text = re.sub(r"[^a-z\s]", "", text) # remove punctuation and numbers
    text = re.sub(r"\s+", " ", text).strip() # remove extra spaces
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['label'], 
    test_size=0.2, random_state=42, stratify=df['label']
)

# Tokenisation
max_words = 10000
max_len = 120
embedding_dim = 100

tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Convert texts to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Convert -1,0,1 labels to 0,1,2 for sparse_categorical_crossentropy
y_train_shift = y_train + 1
y_test_shift = y_test + 1

# Training layers for Bidirectional LSTM
model = Sequential([
    Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len),
    Bidirectional(LSTM(256, return_sequences=True)),
    Dropout(0.5),
    LSTM(128),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    metrics=['accuracy']
)

model.summary()

# Training model with Early Stopping and Model Checkpoint
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint("sentiment_model.h5", monitor='val_loss', save_best_only=True)

history = model.fit(
    X_train_pad, y_train_shift,
    validation_split=0.1,
    epochs=20,
    batch_size=64,
    callbacks=[early_stop, checkpoint]
)

# Save LSTM model
model.save("sentiment_model_full.h5") 

# Evaluation
loss, acc = model.evaluate(X_test_pad, y_test_shift)
print("Test Accuracy:", acc)

# Conf Matrix
y_pred = model.predict(X_test_pad).argmax(axis=1)
print(classification_report(y_test_shift, y_pred, digits=3))
print("Confusion Matrix:\n", confusion_matrix(y_test_shift, y_pred))