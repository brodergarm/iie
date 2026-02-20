import streamlit as st
import pandas as pd
import numpy as np


st.title("INTELLIGENT IDEA ANALYSIS ENGINE")

from sentence_transformers import SentenceTransformer
@st.cache_resource
def load_model():
    print("loading bert model...") # check console
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# load data
if 'df' not in st.session_state:
    df = pd.read_csv("idea_sample.csv")
    vals = df['OriginalText'].tolist()
    vecs = model.encode(vals)
    df['vec'] = list(vecs)
    st.session_state.df = df

# side 
st.sidebar.write("### Settings")
mode = st.sidebar.selectbox("View mode:", ["t-SNE", "PCA"])
st.sidebar.text("Btw: Close dots = similar ideas.")

# check
idea_input = st.text_input("Type your idea here (test it against the database):")

if st.button("Check it") and idea_input:
    u_vec = model.encode([idea_input])[0]
    curr_data = st.session_state.df
    
    
    all_v = np.stack(curr_data['vec'].values)
    # math: dot product / (norm * norm)
    sims = np.dot(all_v, u_vec) / (np.linalg.norm(all_v, axis=1) * np.linalg.norm(u_vec))
    
    score = np.max(sims)
    idx = np.argmax(sims)
    
    st.write(f"Match score: {round(float(score), 4)}")
    
    if score > 0.8:
        st.error("Too similar! This idea already exists.")
        st.info(f"Existing: {curr_data.iloc[idx]['OriginalText']}")
    elif score > 0.6:
        st.warning("Kind of similar")
    else:
        st.success("Unique idea! Added to database.")
        new_item = pd.DataFrame({'OriginalText': [idea_input], 'Category': ['New Idea'], 'vec': [u_vec]})
        st.session_state.df = pd.concat([curr_data, new_item], ignore_index=True)

st.write("---")

# Viz section
import plotly.express as px

d_viz = st.session_state.df
v_matrix = np.stack(d_viz['vec'].values)

if mode == "t-SNE":
    from sklearn.manifold import TSNE
    res = TSNE(n_components=2, perplexity=10).fit_transform(v_matrix)
else:
    from sklearn.decomposition import PCA
    res = PCA(n_components=2).fit_transform(v_matrix)

d_viz['x'] = res[:, 0]
d_viz['y'] = res[:, 1]

# basic scatter plot
fig = px.scatter(d_viz, x='x', y='y', color='Category', hover_data=['OriginalText'])
st.plotly_chart(fig)

with st.expander("Hoe it works)"):
    st.write("Uses S-BERT (MiniLM) to generate semantic embeddings for each entry. "
"Similarity is computed as cosine distance between embedding vectors. "
"The map visualizes clusters using t-SNE or PCA dimensionality reduction.")


