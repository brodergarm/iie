import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# 1. Konfigurasjon
st.set_page_config(page_title="CI Idea Engine PRO", layout="wide")

@st.cache_resource
def load_heavy_model():
    # KRITISK ENDRING: Flerspråklig modell som forstår norsk!
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_heavy_model()

# 2. AKTIVT MINNE
if 'memory' not in st.session_state:
    df = pd.read_csv("idea_sample.csv")[['OriginalText', 'Category']]
    
    with st.spinner('Aktiverer dyp semantisk analyse...'):
        embeddings = model.encode(df['OriginalText'].tolist())
        df['vector'] = list(embeddings)
    
    st.session_state.memory = df

# --- UI ---
st.title("🧠 Intelligent Idea Analysis Engine")
st.markdown("---")

# 3. VALIDERINGS-MOTOREN
st.header("🔍 Validering av Originalitet")

user_input = st.text_input("Skriv inn en idé for å teste systemet:")

if user_input:
    # A. Vektoriser input
    user_vec = model.encode([user_input])[0]
    
    # B. Sammenlign mot minnet
    mem = st.session_state.memory
    similarities = mem['vector'].apply(lambda x: 1 - cosine(user_vec, x))
    
    max_sim = similarities.max()
    best_match = mem.iloc[similarities.idxmax()]
    
    # C. JUSTERTE TERSKLER (viktig for norsk!)
    if max_sim > 0.85:  # Litt lavere terskel siden flerspråklig modell er mindre presis
        st.error(f"❌ **AVVIST:** Dette er et direkte duplikat av noe vi allerede har.")
        st.write(f"**Eksisterende idé:** '{best_match['OriginalText']}'")
        st.write(f"**Likhet:** {max_sim:.1%}")
    elif max_sim > 0.65:  # Justert ned fra 0.75
        st.warning(f"⚠️ **SEMANTISK LIKHET:** Systemet ser at dette betyr det samme som en annen idé.")
        st.write(f"**Konseptuelt likt:** '{best_match['OriginalText']}' (Likhet: {max_sim:.1%})")
    else:
        st.success(f"✅ **GODKJENT:** Ideen er unik og lagret i minnet.")
        new_row = pd.DataFrame({
            'OriginalText': [user_input], 
            'Category': ['Verified Unique'], 
            'vector': [user_vec]
        })
        st.session_state.memory = pd.concat([st.session_state.memory, new_row], ignore_index=True)
    
    # BEVIS
    with st.expander("Teknisk bevis på Computational Intelligence"):
        st.write("Systemet har dekomponert ideen din til en 384-dimensjonal vektor.")
        st.write(f"Høyeste matematiske likhetsskåre funnet: **{max_sim:.3f}**")
        
        # VIS TOPP 5 MATCHES
        st.write("**Topp 5 mest like ideer:**")
        top_5 = similarities.nlargest(5)
        for idx, sim in top_5.items():
            st.write(f"- {mem.iloc[idx]['OriginalText']} → {sim:.1%}")
