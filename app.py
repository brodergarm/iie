import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE

st.set_page_config(page_title="Intelligent Idea Engine PRO", layout="wide")

@st.cache_resource
def load_all():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = pd.read_csv("idea_sample.csv")
    
    # Gjør om vektorer fra tekst til tall
    df['vector'] = df['vector'].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
    
    # HVIS koordinatene mangler, beregn dem her og nå
    if 'x' not in df.columns:
        embeddings = np.stack(df['vector'].values)
        tsne = TSNE(n_components=2, random_state=42)
        coords = tsne.fit_transform(embeddings)
        df['x'], df['y'] = coords[:, 0], coords[:, 1]
        
    return model, df

model, df = load_all()

st.title("🚀 Intelligent Idea Analysis Engine")

# 1. INTERAKTIVT KART (Dette er det avanserte!)
st.header("🌐 Interaktivt Semantisk Kart")
fig = px.scatter(df, x='x', y='y', color='Category', hover_data=['OriginalText'],
                 template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Prism)
st.plotly_chart(fig, use_container_width=True)

# 2. VALIDERINGS-INTERFACE (Ref: Objective 4)
st.divider()
st.header("🔍 Intelligent Validering")
user_input = st.text_input("Skriv inn en idé for å sjekke originalitet:")

if user_input:
    user_vec = model.encode([user_input])[0]
    df['sim'] = df['vector'].apply(lambda x: 1 - cosine(user_vec, x))
    top_match = df.sort_values('sim', ascending=False).iloc[0]
    
    if top_match['sim'] > 0.85:
        st.error(f"⚠️ Duplikat! Likhet: {top_match['sim']:.1%}")
        st.write(f"Ligner på: {top_match['OriginalText']}")
    else:
        st.success(f"✅ Unik idé! Likhet: {top_match['sim']:.1%}")
        st.info(f"Kategori: {top_match['Category']}")
