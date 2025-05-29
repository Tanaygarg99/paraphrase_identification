import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load classifier and Sentence-BERT model
clf = joblib.load("paraphrase_classifier.pkl")
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Streamlit UI
st.title("🧠 Paraphrase Detection App")
st.write("Check if two sentences mean the same thing.")

sentence1 = st.text_input("Enter Sentence 1")
sentence2 = st.text_input("Enter Sentence 2")

if st.button("Check Paraphrase"):
    emb1 = model.encode([sentence1])[0]
    emb2 = model.encode([sentence2])[0]
    sim = cosine_similarity([emb1], [emb2])[0][0]
    prediction = clf.predict([[sim]])[0]

    st.write("### Result:")
    st.success("✅ Paraphrase" if prediction else "❌ Not a Paraphrase")
    st.write("**Similarity Score:**", round(sim, 2))
