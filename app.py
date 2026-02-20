import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

st.set_page_config(page_title="Idea Checker", layout="wide")


@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


model = load_model()

# Data loading
if 'df' not in st.session_state:
    try:
        df = pd.read_csv("idea_sample.csv")
        # Pre-calculating embeddings to save time later
        txt_list = df['OriginalText'].fillna('').tolist()
        vecs = model.encode(txt_list)
        df['vector'] = list(vecs)
        st.session_state.df = df
    except Exception as e:
        st.error(f"Could not load csv: {e}")
        st.stop()

st.title("Idea Similarity Engine")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    method = st.radio("Reduction method:", ["PCA", "t-SNE"])
    threshold = st.slider("Similarity limit", 0.0, 1.0, 0.75)
    if method == "t-SNE":
        perp = st.slider("Perplexity", 5, 50, 25)

# Input area
user_input = st.text_area("Write your idea here:", height=100)

if st.button("Check Originality", type="primary") and user_input:
    u_vec = model.encode([user_input])[0]
    df = st.session_state.df

    # Simple cosine similarity loop
    sims = df['vector'].apply(lambda x: 1 - cosine(u_vec, x))
    max_sim = sims.max()
    best_idx = sims.idxmax()

    st.markdown("---")
    if max_sim > threshold:
        st.error(f"Found match! Similarity: {max_sim:.3f}")
        st.info(f"Existing idea: {df.iloc[best_idx]['OriginalText']}")
    else:
        st.success(f"Looks good! Similarity: {max_sim:.3f}")
        # Add new idea to memory
        new_row = pd.DataFrame({
            'OriginalText': [user_input],
            'Category': ['Verified'],
            'vector': [u_vec]
        })
        st.session_state.df = pd.concat([df, new_row], ignore_index=True)

# Visualization
st.subheader(f"Semantic Map ({method})")

if not st.session_state.df.empty:
    all_vectors = np.stack(st.session_state.df['vector'].values)

    if method == "PCA":
        red = PCA(n_components=2)
        coords = red.fit_transform(all_vectors)
    else:
        # t-SNE is slower, but looks better
        red = TSNE(n_components=2, perplexity=perp, random_state=1)
        coords = red.fit_transform(all_vectors)

    plot_df = st.session_state.df.copy()
    plot_df['x'] = coords[:, 0]
    plot_df['y'] = coords[:, 1]

    fig = px.scatter(
        plot_df, x='x', y='y',
        color='Category',
        hover_data=['OriginalText'],
        template="plotly_dark",
        title=f"Distribution using {method}"
    )
    st.plotly_chart(fig, use_container_width=True)