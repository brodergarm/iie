import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np

# 1. Sideoppsett og introduksjon (Ref: Project Proposal [cite: 18, 21])
st.set_page_config(page_title="Intelligent Idea Analysis Engine", layout="wide")

@st.cache_resource
def load_resources():
    # Laster AI-modellen (Computational Intelligence modellen din [cite: 27])
    model = SentenceTransformer('all-MiniLM-L6-v2')
    try:
        # Laster datasettet (Source of data [cite: 34])
        df = pd.read_csv("idea_sample.csv")
        # Konverterer tekst-vektorer tilbake til tall (numpy arrays)
        df['vector'] = df['vector'].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
        return model, df
    except Exception as e:
        return model, None

model, df_library = load_resources()

st.title("💡 Intelligent Idea Analysis Engine")
st.markdown("""
Dette systemet analyserer brukerinnsendte ideer for å trekke ut konsepter, kategorisere temaer og detektere duplikater basert på semantisk mening[cite: 20, 21].
""")

# --- KAPITTEL 1: VISUALISERING (Ref: Objective 5 ) ---
st.header("📊 Semantisk Landskap")
col1, col2 = st.columns([2, 1])

with col1:
    # Viser t-SNE visualiseringen (Ref: Objective 5 )
    try:
        st.image("Figure_1.png", caption="Visualisering av 100 000 ideer fordelt på 8 sektorer.")
    except:
        st.warning("Bildefilen 'Figure_1.png' ble ikke funnet på GitHub.")

with col2:
    st.write("### Cluster-analyse")
    st.write("""
    Ved bruk av **Computational Intelligence (CI)** grupperes ideer matematisk basert på innhold. 
    Dette gjør det mulig å identifisere temaer og validere unikhet på tvers av 100 000 bidrag[cite: 27, 28].
    """)

# --- KAPITTEL 2: LIVE DEMO (Validation Feature ) ---
st.divider()
st.header("🚀 Prøv systemet selv")
st.write("Skriv inn en idé for å sjekke om den er unik eller et duplikat av et eksisterende konsept.")

user_input = st.text_input("Din idé:")

if user_input:
    if df_library is not None:
        # 1. Generer vektor for brukerens input
        user_vec = model.encode([user_input])[0]
        
        # 2. Beregn likhet (Cosine Similarity) mot biblioteket [cite: 27, 28]
        similarities = df_library['vector'].apply(lambda x: 1 - cosine(user_vec, x))
        max_sim = similarities.max()
        best_match_idx = similarities.idxmax()
        best_match = df_library.iloc[best_match_idx]
        
        # 3. Logikk for duplikatkontroll 
        if max_sim > 0.80: # Terskelverdi for duplikater
            st.error(f"⚠️ **Duplikat oppdaget!** (Likhetsscore: {max_sim:.2%})")
            st.write(f"**Eksisterende idé i systemet:** {best_match['OriginalText']}")
            st.write(f"**Kategori:** {best_match['Category']}")
        else:
            st.success(f"✅ **Ideen er validert som unik!** (Høyeste likhet funnet: {max_sim:.2%})")
            st.info(f"Systemet har kategorisert denne som: **{best_match['Category']}**")
    else:
        st.error("Kunne ikke laste database-eksempel (idea_sample.csv). Vennligst sjekk GitHub-repositoryet.")

# --- PROSJEKTINFO (Ref: Project Proposal [cite: 24, 25]) ---
with st.expander("Om prosjektet og teknologien"):
    st.write("Denne motoren er bygget som en komplett data-pipeline[cite: 24].")
    st.write("Den bruker avansert tekst-preprocessing og Computational Intelligence for å sikre kvalitet i idé-databaser[cite: 25, 27].")
