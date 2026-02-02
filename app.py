import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE

# 1. Konfigurasjon
st.set_page_config(page_title="AI Idea Engine PRO", layout="wide")

# 2. Last modellen (Hjernen)
@st.cache_resource
def load_ai_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_ai_model()

# 3. Initialiser Minnet (Dette gjør den smart og hindrer "abc"-feilen)
if 'idea_db' not in st.session_state:
    # Vi starter med en tom liste eller laster fra CSV hvis den finnes
    try:
        df = pd.read_csv("idea_sample.csv")[['OriginalText', 'Category']]
        # Forhåndsberegn vektorer for start-dataene
        embeddings = model.encode(df['OriginalText'].tolist())
        df['vector'] = list(embeddings)
        st.session_state.idea_db = df
    except:
        st.session_state.idea_db = pd.DataFrame(columns=['OriginalText', 'Category', 'vector'])

# --- UI DESIGN ---
st.title("🧠 Intelligent Idea Analysis Engine")
st.write("Dette er ikke bare tekst-matching. Systemet bruker **Computational Intelligence (CI)** for å forstå meningen bak ordene.")

# --- MODUL 1: TEST INTELLIGENSEN ---
st.header("🔍 Sanntids Validering")
user_input = st.text_input("Skriv inn en idé (Test f.eks. 'abc' to ganger, eller 'flygende bil' og 'luftbårent kjøretøy'):")

if user_input:
    # A. GJØR OM TIL MATEMATIKK (Vektorisering)
    user_vec = model.encode([user_input])[0]
    
    # B. SAMMENLIGN MED ALT I MINNET
    db = st.session_state.idea_db
    
    if not db.empty:
        # Finn likhet (Cosine Similarity)
        similarities = db['vector'].apply(lambda x: 1 - cosine(user_vec, x))
        max_sim = similarities.max()
        best_match = db.iloc[similarities.idxmax()]
    else:
        max_sim = 0
    
    # C. LOGIKK FOR DUPLIKAT (Mål: Reject duplicates )
    if max_sim > 0.98: # Nesten identisk (som "abc")
        st.error(f"❌ **AVVIST:** Dette konseptet eksisterer allerede i motoren.")
        st.write(f"**Treff i minnet:** '{best_match['OriginalText']}' (Likhet: {max_sim:.1%})")
    elif max_sim > 0.80: # Semantisk likt (samme mening)
        st.warning(f"⚠️ **SEMANTISK DUPLIKAT:** Vi har allerede en idé med nesten samme betydning.")
        st.write(f"**Lignende konsept:** '{best_match['OriginalText']}' (Likhet: {max_sim:.1%})")
    else:
        # GODKJENT - Lagre i minnet!
        st.success(f"✅ **GODKJENT:** Ideen er unik og er nå lagret i minnet.")
        new_data = pd.DataFrame({
            'OriginalText': [user_input], 
            'Category': ['User Input'], 
            'vector': [user_vec]
        })
        st.session_state.idea_db = pd.concat([st.session_state.idea_db, new_data], ignore_index=True)
        st.balloons()

    # D. VIS "HJERNEN" (Bevis på CI)
    with st.expander("Se teknisk analyse (Vektor-data)"):
        st.write("Slik ser ideen din ut for AI-modellen (første 10 dimensjoner):")
        st.code(str(user_vec[:10]))
        st.write("Dette er den semantiske signaturen som brukes for å detektere duplikater[cite: 21, 27].")

# --- MODUL 2: VISUALISERING ---
st.divider()
st.header("🌐 Det Semantiske Landskapet")
if st.button("Generer/Oppdater Interaktivt Kart"):
    db = st.session_state.idea_db
    if len(db) > 2:
        embeddings = np.stack(db['vector'].values)
        # Bruker t-SNE for å klynge ideer [cite: 29]
        tsne = TSNE(n_components=2, perplexity=min(30, len(db)-1), random_state=42)
        coords = tsne.fit_transform(embeddings)
        db['x'], db['y'] = coords[:, 0], coords[:, 1]
        
        fig = px.scatter(db, x='x', y='y', color='Category', hover_data=['OriginalText'],
                         title="Idé-klynger (t-SNE)", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Legg inn minst 3 ideer for å generere et kart.")
