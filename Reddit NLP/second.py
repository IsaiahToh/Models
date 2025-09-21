import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from collections import Counter
import numpy as np

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text) # remove URLs
    text = re.sub(r"[^a-z\s]", "", text) # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip() # remove extra spaces
    return text

scraped_posts = []

cleaned_posts = [clean_text(post) for post in scraped_posts]

# NMF model for retrieving topics
# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(cleaned_posts)

# Fit NMF
n_topics = 5
nmf = NMF(n_components=n_topics, random_state=42)
nmf_features = nmf.fit_transform(X_tfidf)

feature_names = tfidf_vectorizer.get_feature_names_out()

# Display top words for each topic
for topic_idx, topic in enumerate(nmf.components_):
    top_words = [feature_names[i] for i in topic.argsort()[-10:][::-1]]
    print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

# Word2Vec + KMeans for trending lingo
# Tokenize posts for Word2Vec
tokenized_posts = [simple_preprocess(post) for post in cleaned_posts]

w2v_model = Word2Vec(
    sentences=tokenized_posts,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1
)

# Extract all word vectors
words = list(w2v_model.wv.index_to_key)
word_vectors = np.array([w2v_model.wv[word] for word in words])

n_clusters = max(2, int(np.sqrt(len(words))))
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(word_vectors)
labels = kmeans.labels_

# Map words to clusters
cluster_to_words = {}
for word, label in zip(words, labels):
    cluster_to_words.setdefault(label, []).append(word)

# Per cluster, rank words by frequency in posts
token_flat = [word for sentence in tokenized_posts for word in sentence]
word_counts = Counter(token_flat)

trending_lingo = []
for cluster_words in cluster_to_words.values():
    sorted_words = sorted(cluster_words, key=lambda w: word_counts[w], reverse=True)
    trending_lingo.extend(sorted_words[:5]) # 5 most frequent per cluster

# Remove duplicates and show
trending_lingo = list(dict.fromkeys(trending_lingo))
print("Top Trending Lingo / Phrases:")
print(", ".join(trending_lingo))
