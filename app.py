import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np

# Sideoppsett
st.set_page_config(page_title="Intelligent Idea Analysis Engine", layout="wide")

@st.cache_resource
def load_all():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    try:
        df = pd.read_csv("idea_sample.csv")
        # Gjør tekst-vektorer om til tall
        df['vector'] = df['vector'].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
        return model, df
    except:
        return model, None

model, df_library = load_all()

# --- INTRODUKSJON ---
st.title("💡 Intelligent Idea Analysis Engine")
st.write("Dette systemet analyserer brukerinnsendte ideer for å trekke ut konsepter og validere originalitet.")

# --- VISUALISERING (Mål: Present results in a clear format) ---
st.header("📊 Semantisk Landskap")
try:
    st.image("Figure_1.png", caption="Visualisering av idé-klynger og temaer.")
except:
    st.error("Kunne ikke laste Figure_1.png. Sjekk at filnavnet er riktig på GitHub.")

# --- LIVE DEMO (Mål: Reject duplicates) ---
st.divider()
st.header("🚀 Prøv systemet selv")
user_input = st.text_input("Skriv inn en idé (f.eks. om naboer som deler verktøy):")

if user_input and df_library is not None:
    user_vec = model.encode([user_input])[0]
    similarities = df_library['vector'].apply(lambda x: 1 - cosine(user_vec, x))
    max_sim = similarities.max()
    best_match = df_library.iloc[similarities.idxmax()]

    if max_sim > 0.82:
        st.error(f"⚠️ Duplikat detektert! (Likhet: {max_sim:.1%})")
        st.write(f"**Ligner på:** {best_match['OriginalText']}")
    else:
        st.success(f"✅ Unik idé! (Høyeste likhet: {max_sim:.1%})")
        st.info(f"Kategori: **{best_match['Category']}**")
