import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# --- App Config ---
st.set_page_config(page_title="CI Idea Engine", layout="wide")

@st.cache_resource
def load_model():
    # MiniLM is fast and good for this
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- State Management ---
if 'df' not in st.session_state:
    df = pd.read_csv("idea_sample.csv")
    txt = df['OriginalText'].fillna('').tolist()
    embeddings = model.encode(txt)
    df['vector'] = list(embeddings)
    st.session_state.df = df
    st.session_state.last_sim = None

# --- UI Sidebar ---
with st.sidebar:
    st.header("Viz Settings")
    method = st.radio("Dim Reduction:", ["PCA", "t-SNE"])
    if method == "t-SNE":
        perp = st.slider("Perplexity", 5, 50, 30)
    
    st.markdown("---")
    st.write(f"Total ideas in DB: {len(st.session_state.df)}")

# --- Main Logic ---
st.title("Idea Analysis & Validation")

user_input = st.text_input("New Idea:", placeholder="Type your idea here...")
btn = st.button("Validate & Analyze", type="primary")

if user_input and btn:
    u_vec = model.encode([user_input])[0]
    df = st.session_state.df
    
    # Calculate similarity
    sims = df['vector'].apply(lambda x: 1 - cosine(u_vec, x))
    max_sim = sims.max()
    match = df.iloc[sims.idxmax()]['OriginalText']
    st.session_state.last_sim = max_sim
    
    col1, col2 = st.columns(2)
    
    with col1:
        if max_sim > 0.85:
            st.error(f"Duplicate! Similarity: {max_sim:.2%}")
            st.info(f"Matches: {match}")
        elif max_sim > 0.65:
            st.warning(f"Very similar! Similarity: {max_sim:.2%}")
            st.write(f"Closest idea: {match}")
        else:
            st.success(f"Unique! Similarity: {max_sim:.2%}")
            # Add to local memory
            new_row = pd.DataFrame({'OriginalText': [user_input], 'Category': ['Verified'], 'vector': [u_vec]})
            st.session_state.df = pd.concat([df, new_row], ignore_index=True)

    with col2:
        with st.expander("Why this works (CI Proof)"):
            st.write(f"1. **Transformer model** (384D vectors)")
            st.write(f"2. **Cosine similarity** check")
            st.write(f"3. **Unsupervised clustering** ({method})")
            if st.session_state.last_sim:
                st.json({"input": user_input, "score": round(max_sim, 3), "status": "Checked"})

# --- The Map ---
st.header("Semantic Landscape")
all_vecs = np.stack(st.session_state.df['vector'].values)

if method == "PCA":
    coords = PCA(n_components=2).fit_transform(all_vecs)
else:
    coords = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(all_vecs)

plot_df = st.session_state.df.copy()
plot_df['x'], plot_df['y'] = coords[:, 0], coords[:, 1]

fig = px.scatter(plot_df, x='x', y='y', color='Category', 
                 hover_data=['OriginalText'], template="plotly_dark",
                 height=600, title=f"Mapped with {method}")
st.plotly_chart(fig, use_container_width=True)

# --- Instructions ---
with st.expander("System Manual"):
    st.write("""
    - **Step 1:** Enter your idea in the text box.
    - **Step 2:** Click 'Validate'. The system checks for semantic matches.
    - **Step 3:** View the map. Proximity = Similarity.
    - **Note:** If the idea is unique, it's added to the local database for future checks.
    """)
