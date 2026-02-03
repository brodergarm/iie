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
# KONFIGURASJON
# =============================================================================
st.set_page_config(page_title="CI Idea Engine PRO", layout="wide")

# Fargepalett for kategorier
CATEGORY_COLORS = {
    'AI & Robotics': '#FF2F92',
    'Cybersecurity': '#00B3A4',
    'Health': '#008CBE',
    'Energy & Environment': '#3A9C3A',
    'Space': '#FFC300',
    'Ocean': '#7B1FA2',
    'Education': '#00695C',
    'Food & Biotech': '#F39C12',
    'Verified Unique': '#E74C3C'  # Nye validerte ideer
}

@st.cache_resource
def load_heavy_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_heavy_model()

# =============================================================================
# INITIALISERING AV MINNE OG VISUALISERING
# =============================================================================
if 'memory' not in st.session_state:
    df = pd.read_csv("idea_sample.csv")[['OriginalText', 'Category']]
    
    with st.spinner('🧠 Aktiverer dyp semantisk analyse...'):
        embeddings = model.encode(df['OriginalText'].tolist())
        df['vector'] = list(embeddings)
    
    st.session_state.memory = df
    st.session_state.new_idea_coords = None
    st.session_state.show_animation = False

# =============================================================================
# DIMENSJONSREDUKSJON FOR VISUALISERING
# =============================================================================
@st.cache_data
def compute_2d_projection(vectors, method='tsne', perplexity=30):
    """Reduser vektorene til 2D for visualisering"""
    vectors_array = np.array(vectors.tolist())
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, perplexity=perplexity, 
                       random_state=42, max_iter=1000)
    else:  # PCA
        reducer = PCA(n_components=2, random_state=42)
    
    coords = reducer.fit_transform(vectors_array)
    return coords

def create_semantic_map(df, new_point=None, highlight_similar=None):
    """Lager interaktivt scatter plot med Plotly"""
    
    # Beregn 2D-koordinater
    coords = compute_2d_projection(df['vector'], method='pca')
    
    fig = go.Figure()
    
    # Plot eksisterende ideer per kategori
    for category in df['Category'].unique():
        mask = df['Category'] == category
        cat_coords = coords[mask]
        
        if len(cat_coords) == 0:
            continue
        
        color = CATEGORY_COLORS.get(category, '#95A5A6')
        
        # Hovering tekst
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
            hovertemplate='<b>%{text}</b><br>Kategori: ' + category + '<extra></extra>'
        ))
    
    # Legg til ny idé med animasjon
    if new_point is not None:
        fig.add_trace(go.Scatter(
            x=[new_point[0]],
            y=[new_point[1]],
            mode='markers+text',
            name='NY IDÉ',
            marker=dict(
                size=20,
                color='#FFD700',
                symbol='star',
                line=dict(width=3, color='white')
            ),
            text=['★'],
            textposition='top center',
            textfont=dict(size=20, color='#FFD700'),
            hovertemplate='<b>DIN NYE IDÉ</b><extra></extra>'
        ))
    
    # Highlight lignende ideer
    if highlight_similar is not None and len(highlight_similar) > 0:
        similar_coords = coords[highlight_similar]
        similar_texts = df.iloc[highlight_similar]['OriginalText'].values
        
        fig.add_trace(go.Scatter(
            x=similar_coords[:, 0],
            y=similar_coords[:, 1],
            mode='markers',
            name='Lignende ideer',
            marker=dict(
                size=15,
                color='red',
                opacity=0.8,
                symbol='circle-open',
                line=dict(width=3, color='red')
            ),
            text=similar_texts,
            hovertemplate='<b>LIGNENDE:</b><br>%{text}<extra></extra>'
        ))
    
    # Layout
    fig.update_layout(
        title={
            'text': '🗺️ Semantisk Idé-Landskap',
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
# UI - HOVEDSEKSJON
# =============================================================================
st.title("🧠 Intelligent Idea Analysis Engine")
st.markdown("### *Semantic validation with live visualization*")
st.markdown("---")

# Sidebar for innstillinger
with st.sidebar:
    st.header("⚙️ Visualisering")
    viz_method = st.selectbox("Dimensjonsreduksjon:", ["PCA", "t-SNE"], index=0)
    
    if viz_method == "t-SNE":
        perplexity = st.slider("t-SNE Perplexity:", 5, 50, 30)
    
    st.markdown("---")
    st.header("📊 Statistikk")
    st.metric("Totalt antall ideer", len(st.session_state.memory))
    
    category_counts = st.session_state.memory['Category'].value_counts()
    st.write("**Ideer per kategori:**")
    for cat, count in category_counts.items():
        st.write(f"• {cat}: {count}")

# =============================================================================
# VALIDERING AV NYE IDEER
# =============================================================================
st.header("🔍 Validering av Originalitet")

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_input("Skriv inn en idé for å teste systemet:", 
                                placeholder="F.eks: App for hundelufting med GPS-tracking")

with col2:
    st.write("")
    st.write("")
    validate_button = st.button("🚀 Valider Idé", type="primary")

if user_input and validate_button:
    # A. Vektoriser input
    user_vec = model.encode([user_input])[0]
    
    # B. Sammenlign mot minnet
    mem = st.session_state.memory
    similarities = mem['vector'].apply(lambda x: 1 - cosine(user_vec, x))
    
    max_sim = similarities.max()
    best_match = mem.iloc[similarities.idxmax()]
    
    # Finn topp 5 lignende
    top_5_indices = similarities.nlargest(5).index.tolist()
    
    # C. VALIDERING
    col_result1, col_result2 = st.columns([1, 1])
    
    with col_result1:
        if max_sim > 0.85:
            st.error(f"❌ **AVVIST:** Dette er et direkte duplikat!")
            st.write(f"**Eksisterende idé:** '{best_match['OriginalText']}'")
            st.write(f"**Likhet:** {max_sim:.1%}")
            show_viz = True
            
        elif max_sim > 0.65:
            st.warning(f"⚠️ **SEMANTISK LIKHET DETEKTERT**")
            st.write(f"**Konseptuelt likt:** '{best_match['OriginalText']}'")
            st.write(f"**Likhet:** {max_sim:.1%}")
            show_viz = True
            
        else:
            st.success(f"✅ **GODKJENT:** Ideen er unik!")
            
            # LEGG TIL I MINNET
            new_row = pd.DataFrame({
                'OriginalText': [user_input], 
                'Category': ['Verified Unique'], 
                'vector': [user_vec]
            })
            st.session_state.memory = pd.concat([st.session_state.memory, new_row], 
                                                  ignore_index=True)
            
            # Marker for visualisering
            st.session_state.show_animation = True
            show_viz = True
        
        # Topp 5 matches
        with st.expander("📊 Topp 5 mest like ideer"):
            for idx in top_5_indices:
                sim = similarities.iloc[idx]
                idea = mem.iloc[idx]['OriginalText']
                category = mem.iloc[idx]['Category']
                st.write(f"**{sim:.1%}** - {idea} `({category})`")
    
    with col_result2:
        st.info("**💡 Hvordan fungerer dette?**")
        st.write("""
        1. Din idé konverteres til en 384-dimensjonal vektor
        2. Vektoren sammenlignes med alle eksisterende ideer
        3. Cosine similarity beregnes (1.0 = identisk)
        4. Ideen godkjennes eller avvises basert på terskelverdi
        """)

# =============================================================================
# VISUALISERING
# =============================================================================
st.markdown("---")
st.header("🗺️ Semantisk Idé-Landskap")

# Beregn koordinater for ny idé hvis den eksisterer
new_point_coords = None
if user_input and validate_button:
    # Vi må re-kjøre projeksjonen med den nye ideen inkludert
    all_vectors = st.session_state.memory['vector']
    coords = compute_2d_projection(all_vectors, 
                                   method=viz_method.lower(), 
                                   perplexity=perplexity if viz_method == "t-SNE" else 30)
    
    # Den siste koordinaten er den nye ideen (hvis den ble lagt til)
    if max_sim <= 0.65:  # Hvis godkjent
        new_point_coords = coords[-1]
        highlight_indices = top_5_indices
    else:
        # Finn hvor den ville ha vært
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

# Lag visualisering
fig = create_semantic_map(st.session_state.memory, 
                         new_point=new_point_coords,
                         highlight_similar=highlight_indices)

st.plotly_chart(fig, use_container_width=True)

# Forklaring
with st.expander("ℹ️ Hvordan lese kartet"):
    st.write("""
    - **Hver prikk** representerer én idé
    - **Farger** indikerer kategori
    - **Nærhet** betyr semantisk likhet (ideer med samme mening ligger nært hverandre)
    - **Gull stjerne** ⭐ = Din nye idé
    - **Røde ringer** = Ideer som er lignende din nye idé
    - **Hover** over prikker for å se idé-teksten
    """)

# =============================================================================
# TEKNISK DOKUMENTASJON (FOR OPPGAVEN)
# =============================================================================
st.markdown("---")
st.header("📋 Teknisk Dokumentasjon")

tab1, tab2, tab3 = st.tabs(["Modell", "Algoritme", "Bevis på CI"])

with tab1:
    st.subheader("🤖 Sentence Transformer Model")
    st.code("""
    Model: paraphrase-multilingual-MiniLM-L12-v2
    - Type: Flerspråklig BERT-basert transformer
    - Embedding dimensjoner: 384
    - Språk: 50+ inkludert norsk
    - Bruksområde: Semantisk søk og duplikatdeteksjon
    """, language="python")

with tab2:
    st.subheader("⚙️ Validerings-Algoritme")
    st.code("""
    1. Input → Embedding (384D vektor)
    2. For hver idé i database:
       - Beregn cosine similarity
    3. Finn høyeste similarity score
    4. Beslutningslogikk:
       - > 0.85: AVVIS (duplikat)
       - > 0.65: ADVAR (semantisk lik)
       - ≤ 0.65: GODKJENN (unik)
    5. Legg til i database hvis godkjent
    6. Oppdater visualisering
    """, language="python")

with tab3:
    st.subheader("✅ Computational Intelligence Bevis")
    st.write("""
    Systemet demonstrerer **Computational Intelligence** gjennom:
    
    1. **Semantisk Forståelse**: Modellen forstår at "hunde-app for turer" og 
       "plattform for å lufte bikkja" er samme konsept
    
    2. **Adaptiv Læring**: Nye ideer legges til i minnet og påvirker fremtidige 
       sammenligninger
    
    3. **Flerdimensjonal Analyse**: 384-dimensjonal vektorrepresentasjon fanger 
       nyansert mening
    
    4. **Unsupervised Learning**: t-SNE/PCA clustering uten forhåndsmerking
    
    5. **Real-time Validering**: Øyeblikkelig sammenligning mot 2000+ ideer
    """)
    
    if user_input:
        st.write("**Eksempel fra siste validering:**")
        st.json({
            "input": user_input,
            "embedding_dim": 384,
            "highest_similarity": f"{max_sim:.3f}",
            "decision": "GODKJENT" if max_sim <= 0.65 else "AVVIST",
            "computational_method": "Cosine Similarity + Transformer Embeddings"
        })

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.caption("💻 Christian Garmann Schjelderup | Exam Project: Intelligent Idea Analysis Engine")
st.caption("🔬 Computational Intelligence Model: Sentence-BERT + Semantic Validation")
