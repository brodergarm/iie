import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# 1. Konfigurasjon
st.set_page_config(page_title="CI Idea Engine PRO", layout="wide")

@st.cache_resource
def load_heavy_model():
    # Vi bruker en modell som er bedre på å forstå dyp mening
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_heavy_model()

# 2. AKTIVT MINNE (Dette hindrer "abc"-problemet)
if 'memory' not in st.session_state:
    # Vi laster inn dine 2000 ideer som et startpunkt
    df = pd.read_csv("idea_sample.csv")[['OriginalText', 'Category']]
    # Vi genererer vektorene PÅ NYTT her for å unngå CSV-feil
    with st.spinner('Aktiverer dyp semantisk analyse...'):
        embeddings = model.encode(df['OriginalText'].tolist())
        df['vector'] = list(embeddings)
    st.session_state.memory = df

# --- UI ---
st.title("🧠 Intelligent Idea Analysis Engine")
st.markdown("---")

# 3. VALIDERINGS-MOTOREN (Dette er den "smarte" biten)
st.header("🔍 Validering av Originalitet")
user_input = st.text_input("Skriv inn en idé for å teste systemet:")

if user_input:
    # A. Vektoriser input (on-the-fly)
    user_vec = model.encode([user_input])[0]
    
    # B. Sammenlign mot minnet
    mem = st.session_state.memory
    # Vi regner ut likheten (1 = identisk, 0 = helt ulikt)
    similarities = mem['vector'].apply(lambda x: 1 - cosine(user_vec, x))
    
    max_sim = similarities.max()
    best_match = mem.iloc[similarities.idxmax()]

    # C. STRENG LOGIKK (Her viser vi intelligensen)
    if max_sim > 0.95: 
        st.error(f"❌ **AVVIST:** Dette er et direkte duplikat av noe vi allerede har.")
        st.write(f"**Eksisterende idé:** '{best_match['OriginalText']}'")
    elif max_sim > 0.75:
        st.warning(f"⚠️ **SEMANTISK LIKHET:** Systemet ser at dette betyr det samme som en annen idé.")
        st.write(f"**Konseptuelt likt:** '{best_match['OriginalText']}' (Likhet: {max_sim:.1%})")
    else:
        st.success(f"✅ **GODKJENT:** Ideen er unik og lagret i minnet.")
        # LEGG TIL I MINNET NÅ
        new_row = pd.DataFrame({
            'OriginalText': [user_input], 
            'Category': ['Verified Unique'], 
            'vector': [user_vec]
        })
        st.session_state.memory = pd.concat([st.session_state.memory, new_row], ignore_index=True)

    # BEVIS FOR RAPPORTEN: Vis hva AI-en faktisk ser
    with st.expander("Teknisk bevis på Computational Intelligence"):
        st.write("Systemet har dekomponert ideen din til en 384-dimensjonal vektor.")
        st.write(f"Høyeste matematiske likhetsskåre funnet: **{max_sim}**")
