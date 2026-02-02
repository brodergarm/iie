import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE

st.set_page_config(page_title="Intelligent Idea Engine PRO", layout="wide")

# 1. Last "Hjernen" og initialiser minnet
@st.cache_resource
def load_engine():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Start med dine eksisterende data (Source of data)
    df = pd.read_csv("idea_sample.csv")[['OriginalText', 'Category']]
    
    # Generer vektorer for start-biblioteket (Computational Intelligence)
    with st.spinner('Initialiserer semantisk minne...'):
        embeddings = model.encode(df['OriginalText'].tolist())
        df['vector'] = list(embeddings)
    return model, df

model, base_df = load_engine()

# 2. Bruk session_state for å huske nye ideer (Simulerer database-lagring)
if 'database' not in st.session_state:
    st.session_state.database = base_df.copy()

# --- HOVEDSIDE ---
st.title("🚀 Intelligent Idea Analysis Engine")
st.markdown("Dette systemet validerer originalitet ved å sammenligne innhold mot alt som tidligere er registrert.")

# --- LIVE DEMO: VALIDERING ---
st.header("🔍 Test systemets intelligens")
user_input = st.text_input("Skriv inn en idé for å sjekke unikhet:")

if user_input:
    # Generer vektor for den nye ideen
    user_vec = model.encode([user_input])[0]
    
    # Sammenlign mot alt i "minnet" (både start-data og det du har skrevet inn før)
    current_db = st.session_state.database
    similarities = current_db['vector'].apply(lambda x: 1 - cosine(user_vec, x))
    
    max_sim = similarities.max()
    best_match_idx = similarities.idxmax()
    best_match_text = current_db.iloc[best_match_idx]['OriginalText']

    # Logikk for duplikatkontroll 
    if max_sim > 0.99: # Nesten 100% likhet (som "abc" vs "abc")
        st.error(f"❌ **AVVIST:** Dette er et direkte duplikat av en eksisterende idé.")
        st.write(f"**Funnet i systemet:** '{best_match_text}'")
    elif max_sim > 0.80: # Semantisk likhet (samme mening, ulike ord)
        st.warning(f"⚠️ **MULIG DUPLIKAT:** En svært lignende idé eksisterer allerede.")
        st.write(f"**Lignende konsept:** '{best_match_text}' (Likhet: {max_sim:.1%})")
    else:
        st.success(f"✅ **GODKJENT:** Ideen er unik og er nå lagret i systemets minne.")
        # LEGG TIL I MINNET (Neste gang vil denne bli avvist som duplikat)
        new_row = pd.DataFrame({
            'OriginalText': [user_input], 
            'Category': ['New Submission'], 
            'vector': [user_vec]
        })
        st.session_state.database = pd.concat([st.session_state.database, new_row], ignore_index=True)

# --- VISUALISERING ---
st.divider()
st.header("🌐 Oppdatert Semantisk Kart")
if st.button("Oppdater kart med nye ideer"):
    with st.spinner('Beregener nye klynger...'):
        db = st.session_state.database
        embeddings = np.stack(db['vector'].values)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(db)-1))
        coords = tsne.fit_transform(embeddings)
        db['x'], db['y'] = coords[:, 0], coords[:, 1]
        
        fig = px.scatter(db, x='x', y='y', color='Category', hover_data=['OriginalText'],
                         template="plotly_dark", title="Semantiske klynger inkludert dine bidrag")
        st.plotly_chart(fig, use_container_width=True)
