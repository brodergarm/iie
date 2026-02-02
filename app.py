import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

st.set_page_config(page_title="Intelligent Idea Engine PRO", layout="wide")

@st.cache_resource
def load_all():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = pd.read_csv("idea_sample.csv")
    df['vector'] = df['vector'].apply(lambda x: np.fromstring(x.strip("[]"), sep=','))
    return model, df

model, df = load_all()

# --- SIDEBAR: Analytics ---
st.sidebar.title("📊 Database Analytics")
cat_counts = df['Category'].value_counts()
st.sidebar.bar_chart(cat_counts)
st.sidebar.write(f"Total ideer i analyse: {len(df)}")

# --- HOVEDSIDE ---
st.title("🚀 Intelligent Idea Analysis Engine")
st.markdown("---")

# 1. INTERAKTIVT KART (Dette er det avanserte!)
st.header("🌐 Interaktivt Semantisk Kart")
st.write("Utforsk hvordan ideene klynger seg sammen basert på semantisk mening.")

fig = px.scatter(df, x='x', y='y', color='Category', hover_data=['OriginalText'],
                 title="Semantic Clusters (t-SNE)", template="plotly_dark",
                 color_discrete_sequence=px.colors.qualitative.Prism)
fig.update_traces(marker=dict(size=8, opacity=0.7))
st.plotly_chart(fig, use_container_width=True)

# 2. AVANSERT VALIDERINGS-INTERFACE
st.divider()
st.header("🔍 Intelligent Validering")
user_input = st.text_input("Test en ny idé for originalitet og kategori:")

if user_input:
    user_vec = model.encode([user_input])[0]
    
    # Finn topp 3 treff
    df['sim'] = df['vector'].apply(lambda x: 1 - cosine(user_vec, x))
    top_matches = df.sort_values('sim', ascending=False).head(3)
    
    max_sim = top_matches['sim'].iloc[0]
    
    if max_sim > 0.85:
        st.error(f"⚠️ **Kritisk Duplikat!** Likhet: {max_sim:.1%}")
    elif max_sim > 0.70:
        st.warning(f"🔔 **Potensiell Likhet Funnet.** Likhet: {max_sim:.1%}")
    else:
        st.success(f"✅ **Unik Idé Validert!** Høyeste likhet: {max_sim:.1%}")

    # Vis de 3 nærmeste konseptene
    st.write("### Nærmeste konsepter i databasen:")
    for i, row in top_matches.iterrows():
        st.info(f"**Likhet: {row['sim']:.1%}** | Kategori: {row['Category']}\n\n*\"{row['OriginalText']}\"*")
