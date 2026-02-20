import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

st.set_page_config(page_title="Idea Engine", layout="wide")

# -- Logic & Model --
@st.cache_resource
def load_engine():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_engine()

def get_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

if 'db' not in st.session_state:
    df = pd.read_csv("idea_sample.csv")
    vectors = model.encode(df['OriginalText'].astype(str).tolist())
    df['vec'] = list(vectors)
    st.session_state.db = df

# -- UI --
st.title("Semantic Idea Analysis")

with st.sidebar:
    st.header("Controls")
    viz_type = st.radio("Projection Method:", ["t-SNE", "PCA"])
    st.markdown("---")
    st.write("**How to read the map:**")
    st.caption("Dots represent ideas. Proximity = Semantic similarity. Clusters show groups of related concepts.")

# Input
user_idea = st.text_input("Test your idea here:")
if st.button("Analyze") and user_idea:
    u_vec = model.encode([user_idea])[0]
    db = st.session_state.db
    
    scores = [get_sim(u_vec, x) for x in db['vec']]
    max_score = max(scores)
    
    if max_score > 0.75:
        st.error(f"Red flag: High similarity ({max_score:.3f})")
        st.write(f"Match: {db.iloc[np.argmax(scores)]['OriginalText']}")
    else:
        st.success(f"Unique idea (Score: {max_score:.3f})")
        new_row = pd.DataFrame({'OriginalText': [user_idea], 'Category': ['New'], 'vec': [u_vec]})
        st.session_state.db = pd.concat([db, new_row], ignore_index=True)

# -- Mapping --
st.subheader(f"Semantic Landscape ({viz_type})")
all_vecs = np.array(st.session_state.db['vec'].tolist())

if viz_type == "t-SNE":
    reducer = TSNE(n_components=2, perplexity=20, random_state=42)
else:
    reducer = PCA(n_components=2)

coords = reducer.fit_transform(all_vecs)
plot_df = st.session_state.db.copy()
plot_df['x'], plot_df['y'] = coords[:, 0], coords[:, 1]

fig = px.scatter(plot_df, x='x', y='y', color='Category', hover_name='OriginalText', template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# -- Brief Technical Proof --
with st.expander("Technical Background"):
    st.write("""
    **Why this works:**
    The system uses a Transformer-based model (S-BERT) to map text into a 384-dimensional space. 
    By calculating the angle between vectors (Cosine Similarity), we identify conceptual duplicates 
    even if the wording is different. t-SNE/PCA then projects these dimensions down to 2D for visualization.
    """)
