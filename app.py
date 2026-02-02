import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np

# 1. Bruk nøyaktig samme modell som i SQL-scriptet ditt
@st.cache_resource
def load_engine():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Vi laster inn din "hjerne" (idea_sample.csv)
    df = pd.read_csv("idea_sample.csv")
    
    # VIKTIG: Gjør om tekst-strengen i CSV-en tilbake til ekte tall
    # Dette er ofte her det feiler og gjør den "dum"
    df['vector'] = df['vector'].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
    return model, df

model, df_library = load_engine()

# --- DIN ORIGINALE LOGIKK ---
user_input = st.text_input("Test din idé her:")

if user_input:
    # Generer vektor på nøyaktig samme måte som i SQL-motoren
    user_vec = model.encode([user_input])[0]
    
    # Finn likhet (Cosine Similarity)
    # Dette er "hjernen" som ser at "verktøy" = "utstyr"
    similarities = df_library['vector'].apply(lambda x: 1 - cosine(user_vec, x))
    
    index_of_best = similarities.idxmax()
    max_score = similarities.max()
    best_match = df_library.iloc[index_of_best]

    # Terskelen for å avvise duplikater (Justert for bedre presisjon)
    if max_score > 0.82: 
        st.error(f"⚠️ **Duplikat detektert!** Likhet: {max_score:.1%}")
        st.info(f"Denne ligner for mye på: '{best_match['OriginalText']}'")
    else:
        st.success(f"✅ **Unik idé!** Høyeste likhet var bare {max_score:.1%}")
        # Bruk kategori-logikken fra din fungerende motor
        st.write(f"Kategorisert som: **{best_match['Category']}**")
