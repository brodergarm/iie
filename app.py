import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import plotly.express as px
from sklearn.manifold import TSNE

st.set_page_config(page_title="Idea Engine v1", layout="wide")
st.title("Idea Similarity Tool")

# Quick helper for similarity - manually written to avoid extra imports
def get_cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

@st.cache_resource
def load_model():
    # Using MiniLM, it's fast enough for this
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load data into session
if 'main_df' not in st.session_state:
    df = pd.read_csv("idea_sample.csv")
    # need to convert to list for the encoder
    texts = df['OriginalText'].astype(str).tolist()
    vectors = model.encode(texts)
    df['vec'] = list(vectors)
    st.session_state.main_df = df

# Input section
new_idea = st.text_input("New idea description:")
run_check = st.button("Check Similarity")

if run_check and new_idea:
    u_vec = model.encode([new_idea])[0]
    data = st.session_state.main_df
    
    # Check all existing ideas
    scores = [get_cosine_sim(u_vec, x) for x in data['vec']]
    top_score = max(scores)
    best_match_idx = np.argmax(scores)
    
    st.write(f"Match score: {top_score:.4f}")
    
    if top_score > 0.75:
        st.error("Too similar to an existing concept!")
        st.info(f"Existing idea: {data.iloc[best_idx]['OriginalText']}")
    else:
        st.success("Idea seems unique.")
        # add to memory
        new_row = pd.DataFrame({
            'OriginalText': [new_idea], 
            'Category': ['User Input'], 
            'vec': [u_vec]
        })
        st.session_state.main_df = pd.concat([data, new_row], ignore_index=True)

# Viz part - sticking with t-SNE
st.markdown("---")
st.subheader("Semantic Map")

current_df = st.session_state.main_df
all_vecs = np.array(current_df['vec'].tolist())

# t-SNE logic
tsne = TSNE(n_components=2, perplexity=20, random_state=1)
coords = tsne.fit_transform(all_vecs)

plot_df = current_df.copy()
plot_df['x'] = coords[:, 0]
plot_df['y'] = coords[:, 1]

# Using standard Plotly Express
fig = px.scatter(
    plot_df, x='x', y='y', 
    color='Category', 
    hover_name='OriginalText',
    template="plotly_dark"
)
st.plotly_chart(fig, use_container_width=True)

# Technical notes at the bottom
st.markdown("---")
st.caption("Technical Info: S-BERT embeddings (384D) | Numpy Cosine Similarity | t-SNE Projection")
