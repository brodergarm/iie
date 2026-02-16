import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
st.set_page_config(page_title="CI Idea Engine PRO", layout="wide")

# Color palette for categories
CATEGORY_COLORS = {
    'AI & Robotics': '#FF2F92',
    'Cybersecurity': '#00B3A4',
    'Health': '#008CBE',
    'Energy & Environment': '#3A9C3A',
    'Space': '#FFC300',
    'Ocean': '#7B1FA2',
    'Education': '#00695C',
    'Food & Biotech': '#F39C12',
    'Verified Unique': '#E74C3C'
}

@st.cache_resource
def load_heavy_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_heavy_model()

# =============================================================================
# MEMORY AND VISUALIZATION INITIALIZATION
# =============================================================================
if 'memory' not in st.session_state:
    df = pd.read_csv("idea_sample.csv")[['OriginalText', 'Category']]
    
    with st.spinner('🧠 Activating deep semantic analysis...'):
        embeddings = model.encode(df['OriginalText'].tolist())
        df['vector'] = list(embeddings)
    
    st.session_state.memory = df
    st.session_state.new_idea_coords = None
    st.session_state.show_animation = False

# =============================================================================
# DIMENSIONALITY REDUCTION FOR VISUALIZATION
# =============================================================================
@st.cache_data
def compute_2d_projection(vectors, method='tsne', perplexity=30):
    vectors_array = np.array(vectors.tolist())
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=perplexity, 
                       random_state=42, max_iter=1000)
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    coords = reducer.fit_transform(vectors_array)
    return coords

def create_semantic_map(df, new_point=None, highlight_similar=None):
    coords = compute_2d_projection(df['vector'], method='pca')
    
    fig = go.Figure()
    
    for category in df['Category'].unique():
        mask = df['Category'] == category
        cat_coords = coords[mask]
        
        if len(cat_coords) == 0:
            continue
        
        color = CATEGORY_COLORS.get(category, '#95A5A6')
        hover_text = df[mask]['OriginalText'].values
        
        fig.add_trace(go.Scatter(
            x=cat_coords[:, 0],
            y=cat_coords[:, 1],
            mode='markers',
            name=category,
            marker=dict(
                size=8,
                color=color,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=hover_text,
            hovertemplate='<b>%{text}</b><br>Category: ' + category + '<extra></extra>'
        ))
    
    if new_point is not None:
        fig.add_trace(go.Scatter(
            x=[new_point[0]],
            y=[new_point[1]],
            mode='markers+text',
            name='NEW IDEA',
            marker=dict(
                size=20,
                color='#FFD700',
                symbol='star',
                line=dict(width=3, color='white')
            ),
            text=['★'],
            textposition='top center',
            textfont=dict(size=20, color='#FFD700'),
            hovertemplate='<b>YOUR NEW IDEA</b><extra></extra>'
        ))
    
    if highlight_similar is not None and len(highlight_similar) > 0:
        similar_coords = coords[highlight_similar]
        similar_texts = df.iloc[highlight_similar]['OriginalText'].values
        
        fig.add_trace(go.Scatter(
            x=similar_coords[:, 0],
            y=similar_coords[:, 1],
            mode='markers',
            name='Similar ideas',
            marker=dict(
                size=15,
                color='red',
                opacity=0.8,
                symbol='circle-open',
                line=dict(width=3, color='red')
            ),
            text=similar_texts,
            hovertemplate='<b>SIMILAR:</b><br>%{text}<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': '🗺️ Semantic Idea Landscape',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': 'white'}
        },
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(color='white'),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        hovermode='closest',
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1
        )
    )
    
    return fig

# =============================================================================
# UI - MAIN SECTION
# =============================================================================
st.title("🧠 Intelligent Idea Analysis Engine")
st.markdown("### *Semantic validation with live visualization*")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Visualization")
    viz_method = st.selectbox("Dimensionality reduction:", ["PCA", "t-SNE"], index=0)
    
    if viz_method == "t-SNE":
        perplexity = st.slider("t-SNE Perplexity:", 5, 50, 30)
    
    st.markdown("---")
    st.header("📊 Statistics")
    st.metric("Total ideas", len(st.session_state.memory))
    
    category_counts = st.session_state.memory['Category'].value_counts()
    st.write("**Ideas per category:**")
    for cat, count in category_counts.items():
        st.write(f"• {cat}: {count}")

# =============================================================================
# VALIDATION OF NEW IDEAS
# =============================================================================
st.header("🔍 Originality Validation")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_input("Enter an idea to test the system:", 
                                placeholder="E.g: Dog walking app with GPS tracking")

with col2:
    st.write("")
    st.write("")
    validate_button = st.button("🚀 Validate Idea", type="primary")

if user_input and validate_button:
    user_vec = model.encode([user_input])[0]
    
    mem = st.session_state.memory
    similarities = mem['vector'].apply(lambda x: 1 - cosine(user_vec, x))
    
    max_sim = similarities.max()
    best_match = mem.iloc[similarities.idxmax()]
    
    top_5_indices = similarities.nlargest(5).index.tolist()
    
    col_result1, col_result2 = st.columns([1, 1])
    
    with col_result1:
        if max_sim > 0.85:
            st.error(f"❌ **REJECTED:** This is a direct duplicate!")
            st.write(f"**Existing idea:** '{best_match['OriginalText']}'")
            st.write(f"**Similarity:** {max_sim:.1%}")
            show_viz = True
            
        elif max_sim > 0.65:
            st.warning(f"⚠️ **SEMANTIC SIMILARITY DETECTED**")
            st.write(f"**Conceptually similar to:** '{best_match['OriginalText']}'")
            st.write(f"**Similarity:** {max_sim:.1%}")
            show_viz = True
            
        else:
            st.success(f"✅ **APPROVED:** The idea is unique!")
            
            new_row = pd.DataFrame({
                'OriginalText': [user_input], 
                'Category': ['Verified Unique'], 
                'vector': [user_vec]
            })
            st.session_state.memory = pd.concat([st.session_state.memory, new_row], 
                                                  ignore_index=True)
            
            st.session_state.show_animation = True
            show_viz = True
        
        with st.expander("📊 Top 5 most similar ideas"):
            for idx in top_5_indices:
                sim = similarities.iloc[idx]
                idea = mem.iloc[idx]['OriginalText']
                category = mem.iloc[idx]['Category']
                st.write(f"**{sim:.1%}** - {idea} `({category})`")
    
    with col_result2:
        st.info("**💡 How does this work?**")
        st.write("""
        1. Your idea is converted to a 384-dimensional vector
        2. The vector is compared with all existing ideas
        3. Cosine similarity is calculated (1.0 = identical)
        4. The idea is approved or rejected based on threshold value
        """)

# =============================================================================
# VISUALIZATION
# =============================================================================
st.markdown("---")
st.header("🗺️ Semantic Idea Landscape")

new_point_coords = None
if user_input and validate_button:
    all_vectors = st.session_state.memory['vector']
    coords = compute_2d_projection(all_vectors, 
                                   method=viz_method.lower(), 
                                   perplexity=perplexity if viz_method == "t-SNE" else 30)
    
    if max_sim <= 0.65:
        new_point_coords = coords[-1]
        highlight_indices = top_5_indices
    else:
        temp_df = pd.concat([st.session_state.memory, 
                            pd.DataFrame({'vector': [user_vec]})], 
                           ignore_index=True)
        temp_coords = compute_2d_projection(temp_df['vector'], 
                                           method=viz_method.lower(),
                                           perplexity=perplexity if viz_method == "t-SNE" else 30)
        new_point_coords = temp_coords[-1]
        highlight_indices = top_5_indices
else:
    highlight_indices = None

fig = create_semantic_map(st.session_state.memory, 
                         new_point=new_point_coords,
                         highlight_similar=highlight_indices)

st.plotly_chart(fig, use_container_width=True)

with st.expander("ℹ️ How to read the map"):
    st.write("""
    - **Each dot** represents one idea
    - **Colors** indicate category
    - **Proximity** means semantic similarity (ideas with similar meaning are close together)
    - **Gold star** ⭐ = Your new idea
    - **Red circles** = Ideas similar to your new idea
    - **Hover** over dots to see the idea text
    """)

# =============================================================================
# TECHNICAL DOCUMENTATION
# =============================================================================
st.markdown("---")
st.header("📋 Technical Documentation")

tab1, tab2, tab3 = st.tabs(["Model", "Algorithm", "CI Evidence"])

with tab1:
    st.subheader("🤖 Sentence Transformer Model")
    st.code("""
    Model: paraphrase-multilingual-MiniLM-L12-v2
    - Type: Multilingual BERT-based transformer
    - Embedding dimensions: 384
    - Languages: 50+ including English
    - Use case: Semantic search and duplicate detection
    """, language="python")

with tab2:
    st.subheader("⚙️ Validation Algorithm")
    st.code("""
    1. Input → Embedding (384D vector)
    2. For each idea in database:
       - Calculate cosine similarity
    3. Find highest similarity score
    4. Decision logic:
       - > 0.85: REJECT (duplicate)
       - > 0.65: WARN (semantically similar)
       - ≤ 0.65: APPROVE (unique)
    5. Add to database if approved
    6. Update visualization
    """, language="python")

with tab3:
    st.subheader("✅ Computational Intelligence Evidence")
    st.write("""
    The system demonstrates **Computational Intelligence** through:
    
    1. **Semantic Understanding**: The model understands that "dog walking app" and 
       "platform for pet exercise tracking" are the same concept
    
    2. **Adaptive Learning**: New ideas are added to memory and influence future 
       comparisons
    
    3. **Multidimensional Analysis**: 384-dimensional vector representation captures 
       nuanced meaning
    
    4. **Unsupervised Learning**: t-SNE/PCA clustering without pre-labeling
    
    5. **Real-time Validation**: Instant comparison against 2000+ ideas
    """)
    
    if user_input:
        st.write("**Example from last validation:**")
        st.json({
            "input": user_input,
            "embedding_dim": 384,
            "highest_similarity": f"{max_sim:.3f}",
            "decision": "APPROVED" if max_sim <= 0.65 else "REJECTED",
            "computational_method": "Cosine Similarity + Transformer Embeddings"
        })

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption("💻 Christian Garmann Schjelderup | Exam Project: Intelligent Idea Analysis Engine")
st.caption("🔬 Computational Intelligence Model: Sentence-BERT + Semantic Validation")
