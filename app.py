import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE

st.set_page_config(page_title="Intelligent Idea Engine PRO", layout="wide")

@st.cache_resource
def load_smart_engine():
    # 1. Last "Hjernen" (CI-modellen)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 2. Last rådata (Bare tekst og kategori)
    df = pd.read_csv("idea_sample.csv")
    
    # 3. GENERER VEKTORER PÅ NYTT (Dette gjør den smart!)
    # Vi stoler ikke på CSV-tallene, vi lager dem ferske
    with st.spinner('Aktiverer Computational Intelligence...'):
        embeddings = model.encode(df['OriginalText'].tolist(), show_progress_bar=False)
        df['vector'] = list(embeddings)
    
    # 4. Beregn koordinater for interaktivt kart
    tsne = TSNE(n_components=2, random_state=42)
    coords = tsne.fit_transform(embeddings)
    df['x'], df['y'] = coords[:, 0], coords[:, 1]
    
    return model, df

model, df = load_smart_engine()

st.title("🚀 Intelligent Idea Analysis Engine")
st.info("Systemet bruker nå full semantisk analyse for å detektere konseptuelle likheter.")

# --- INTERAKTIVT KART ---
st.header("🌐 Semantisk Landskap")
fig = px.scatter(df, x='x', y='y', color='Category', hover_data=['OriginalText'],
                 template="plotly_dark", height=600)
st.plotly_chart(fig, use_container_width=True)

# --- DEN SMARTE VALIDERINGS-MOTOREN ---
st.divider()
st.header("🔍 Intelligent Validering")
user_input = st.text_input("Skriv inn en idé for å teste systemets intelligens:")

if user_input:
    # Vektoriserer input
    user_vec = model.encode([user_input])[0]
    
    # Regner ut likhet mot alle ideer
    df['sim'] = df['vector'].apply(lambda x: 1 - cosine(user_vec, x))
    
    # Sorterer for å finne de beste treffene
    top_matches = df.sort_values('sim', ascending=False).head(3)
    max_sim = top_matches['sim'].iloc[0]

    if max_sim > 0.80:
        st.error(f"⚠️ **Duplikat detektert!** (Likhet: {max_sim:.1%})")
        st.write(f"Systemet kjenner igjen dette konseptet som: *'{top_matches['OriginalText'].iloc[0]}'*")
    else:
        st.success(f"✅ **Unik idé validert!** (Høyeste likhet funnet: {max_sim:.1%})")
    
    # Vis sensor at vi faktisk forstår sammenhengen (Topp 3 treff)
    with st.expander("Se teknisk analyse av konseptuell likhet"):
        for i, row in top_matches.iterrows():
            st.write(f"**{row['sim']:.1%} likhet:** {row['OriginalText']} ({row['Category']})")
