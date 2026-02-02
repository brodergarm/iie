import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np

# Sideoppsett
st.title("💡 Intelligent Idea Analysis Engine")

@st.cache_resource
def load_all():
    # 1. Last modellen (CI-modellen) 
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # 2. Last datasettet (Vårt "Library" av ideer) [cite: 30]
    try:
        df = pd.read_csv("idea_sample.csv")
        # Gjør tekst om til tall igjen
        df['vector'] = df['vector'].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
        return model, df
    except:
        return model, None

model, df_library = load_all()

# --- LIVE DEMO ---
st.header("🚀 Valider din idé")
user_input = st.text_input("Skriv inn din idé her:")

if user_input and df_library is not None:
    # Vektoriserer input for å forstå semantisk mening [cite: 21]
    user_vec = model.encode([user_input])[0]
    
    # Sammenligner med eksisterende ideer (Cosine Similarity) [cite: 28]
    similarities = df_library['vector'].apply(lambda x: 1 - cosine(user_vec, x))
    max_sim = similarities.max()
    best_match = df_library.iloc[similarities.idxmax()]

    if max_sim > 0.85:
        st.error(f"⚠️ **Duplikat funnet!** (Likhet: {max_sim:.1%})")
        st.write(f"Dette ligner for mye på: *{best_match['OriginalText']}*")
    else:
        st.success(f"✅ **Unik idé!** (Høyeste likhet funnet: {max_sim:.1%})")
        st.info(f"Systemet kategoriserer denne som: **{best_match['Category']}**")
