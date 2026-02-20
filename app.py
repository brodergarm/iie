import streamlit as st
import pandas as pd
import numpy as np

# simple title
st.title("Idea Similarity Engine")

# Lazy loading the model only when we actually need it
from sentence_transformers import SentenceTransformer
@st.cache_resource
def load_model():
    print("loading bert model...") # check console
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# load data
if 'df' not in st.session_state:
    # hope the file is in the same folder...
    df = pd.read_csv("idea_sample.csv")
    # need to encode everything once 
    vals = df['OriginalText'].tolist()
    vecs = model.encode(vals)
    df['vec'] = list(vecs)
    st.session_state.df = df

# side stuff
st.sidebar.write("### Settings")
mode = st.sidebar.selectbox("View mode:", ["t-SNE", "PCA"])
st.sidebar.text("Btw: Close dots = similar ideas.")

# The check
idea_input = st.text_input("Type your idea here (test it against the database):")

if st.button("Check it") and idea_input:
    # get vector for new input
    u_vec = model.encode([idea_input])[0]
    curr_data = st.session_state.df
    
    # manual similarity calc because i dont want to import scipy
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
        st.warning("Kind of similar. Maybe tweak it a bit?")
    else:
        st.success("Unique idea! Added to local list.")
        # just append it
        new_item = pd.DataFrame({'OriginalText': [idea_input], 'Category': ['New Idea'], 'vec': [u_vec]})
        st.session_state.df = pd.concat([curr_data, new_item], ignore_index=True)

st.write("---")

# Viz section - importing here because its only used for the plot
import plotly.express as px

d_viz = st.session_state.df
v_matrix = np.stack(d_viz['vec'].values)

if mode == "t-SNE":
    from sklearn.manifold import TSNE
    # lowering perplexity so it doesnt crash on small datasets
    res = TSNE(n_components=2, perplexity=10).fit_transform(v_matrix)
else:
    from sklearn.decomposition import PCA
    res = PCA(n_components=2).fit_transform(v_matrix)

d_viz['x'] = res[:, 0]
d_viz['y'] = res[:, 1]

# basic scatter plot
fig = px.scatter(d_viz, x='x', y='y', color='Category', hover_data=['OriginalText'])
st.plotly_chart(fig)

# quick proof for the exam/teacher
with st.expander("Technical stuff (how it works)"):
    st.write("Using S-BERT (MiniLM) for embeddings. Similarity is just cosine distance between vectors. "
             "The map uses t-SNE or PCA to show the clusters.")
