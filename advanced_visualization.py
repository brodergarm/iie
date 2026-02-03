"""
Advanced Visualization Script for Intelligent Idea Analysis Engine
Generates high-quality static and interactive visualizations for the exam report

Author: Christian Garmann Schjelderup
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
SAMPLE_SIZE = 2000  # Use all ideas or sample for faster processing
VIZ_METHOD = 'pca'  # 'pca' or 'tsne'
OUTPUT_PREFIX = f'semantic_map_{VIZ_METHOD}_'

# Color palette
CATEGORY_COLORS = {
    'AI & Robotics': '#FF2F92',
    'Cybersecurity': '#00B3A4',
    'Health': '#008CBE',
    'Energy & Environment': '#3A9C3A',
    'Space': '#FFC300',
    'Ocean': '#7B1FA2',
    'Education': '#00695C',
    'Food & Biotech': '#F39C12',
}

print("=" * 80)
print("INTELLIGENT IDEA ANALYSIS ENGINE - VISUALIZATION GENERATOR")
print("=" * 80)

# ==============================================================================
# STEP 1: LOAD DATA
# ==============================================================================
print("\n[1/6] Loading ideas from CSV...")
df = pd.read_csv("idea_sample.csv")
print(f"   ✓ Loaded {len(df)} ideas across {df['Category'].nunique()} categories")

# Sample if needed for faster processing
if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
    df = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"   ✓ Sampled {SAMPLE_SIZE} ideas for visualization")

# ==============================================================================
# STEP 2: GENERATE EMBEDDINGS
# ==============================================================================
print("\n[2/6] Generating semantic embeddings (this may take a few minutes)...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

embeddings = []
for text in tqdm(df['OriginalText'], desc="   Encoding ideas"):
    embeddings.append(model.encode(text))

embeddings = np.array(embeddings)
print(f"   ✓ Generated {embeddings.shape[0]} embeddings with {embeddings.shape[1]} dimensions")

# ==============================================================================
# STEP 3: DIMENSIONALITY REDUCTION
# ==============================================================================
print(f"\n[3/6] Reducing dimensions using {VIZ_METHOD.upper()}...")

if VIZ_METHOD == 'pca':
    reducer = PCA(n_components=2, random_state=42)
    coords_2d = reducer.fit_transform(embeddings)
    explained_var = reducer.explained_variance_ratio_
    print(f"   ✓ PCA complete - Variance explained: {explained_var[0]:.2%} + {explained_var[1]:.2%} = {sum(explained_var):.2%}")
else:  # t-SNE
    reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, verbose=1)
    coords_2d = reducer.fit_transform(embeddings)
    print(f"   ✓ t-SNE complete")

df['x'] = coords_2d[:, 0]
df['y'] = coords_2d[:, 1]

# ==============================================================================
# STEP 4: STATIC VISUALIZATION (for PDF report)
# ==============================================================================
print("\n[4/6] Creating static visualization...")

plt.figure(figsize=(16, 10), facecolor='#0E1117')
ax = plt.gca()
ax.set_facecolor('#0E1117')

# Plot each category
for category in df['Category'].unique():
    mask = df['Category'] == category
    color = CATEGORY_COLORS.get(category, '#95A5A6')
    
    plt.scatter(
        df[mask]['x'], 
        df[mask]['y'],
        c=color,
        label=category,
        alpha=0.7,
        s=50,
        edgecolors='white',
        linewidth=0.5
    )

plt.title('Semantic Landscape of Ideas', 
          fontsize=24, color='white', pad=20, fontweight='bold')
plt.xlabel(f'{VIZ_METHOD.upper()} Component 1', fontsize=14, color='white')
plt.ylabel(f'{VIZ_METHOD.upper()} Component 2', fontsize=14, color='white')
plt.legend(loc='upper right', framealpha=0.9, fontsize=10, facecolor='#1E1E1E', edgecolor='white')
plt.tick_params(colors='white')
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')

# Save static image
static_filename = f'static_{OUTPUT_PREFIX}{len(df)}.png'
plt.savefig(static_filename, dpi=300, bbox_inches='tight', facecolor='#0E1117')
print(f"   ✓ Static visualization saved: {static_filename}")
plt.close()

# ==============================================================================
# STEP 5: INTERACTIVE VISUALIZATION
# ==============================================================================
print("\n[5/6] Creating interactive visualization...")

fig = go.Figure()

# Add traces for each category
for category in sorted(df['Category'].unique()):
    mask = df['Category'] == category
    category_df = df[mask]
    
    color = CATEGORY_COLORS.get(category, '#95A5A6')
    
    fig.add_trace(go.Scatter(
        x=category_df['x'],
        y=category_df['y'],
        mode='markers',
        name=category,
        marker=dict(
            size=8,
            color=color,
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        text=category_df['OriginalText'],
        hovertemplate='<b>%{text}</b><br>Category: ' + category + '<extra></extra>',
        showlegend=True
    ))

# Update layout
fig.update_layout(
    title={
        'text': '🗺️ Interactive Semantic Landscape',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 28, 'color': 'white'}
    },
    plot_bgcolor='#0E1117',
    paper_bgcolor='#0E1117',
    font=dict(color='white', size=12),
    xaxis=dict(
        title=f'{VIZ_METHOD.upper()} Component 1',
        showgrid=True,
        gridcolor='#333333',
        zeroline=False
    ),
    yaxis=dict(
        title=f'{VIZ_METHOD.upper()} Component 2',
        showgrid=True,
        gridcolor='#333333',
        zeroline=False
    ),
    height=700,
    hovermode='closest',
    legend=dict(
        bgcolor='rgba(30,30,30,0.8)',
        bordercolor='white',
        borderwidth=1,
        font=dict(size=11)
    )
)

# Save interactive HTML
interactive_filename = f'interactive_{OUTPUT_PREFIX}{len(df)}.html'
fig.write_html(interactive_filename)
print(f"   ✓ Interactive visualization saved: {interactive_filename}")

# ==============================================================================
# STEP 6: ANIMATED VISUALIZATION (showing ideas being added)
# ==============================================================================
print("\n[6/6] Creating animated visualization...")

# Create animation frames - showing ideas being added progressively
frames = []
step_size = max(50, len(df) // 40)  # Show ~40 frames

for i in range(step_size, len(df) + 1, step_size):
    frame_data = []
    subset_df = df.iloc[:i]
    
    for category in sorted(df['Category'].unique()):
        mask = subset_df['Category'] == category
        category_df = subset_df[mask]
        
        if len(category_df) == 0:
            continue
        
        color = CATEGORY_COLORS.get(category, '#95A5A6')
        
        frame_data.append(go.Scatter(
            x=category_df['x'],
            y=category_df['y'],
            mode='markers',
            name=category,
            marker=dict(
                size=8,
                color=color,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            text=category_df['OriginalText'],
            hovertemplate='<b>%{text}</b><br>Category: ' + category + '<extra></extra>',
            showlegend=True
        ))
    
    frames.append(go.Frame(data=frame_data, name=str(i)))

# Create initial figure
initial_fig = go.Figure(data=frames[0].data if frames else [])

# Add frames
initial_fig.frames = frames

# Add play/pause buttons
initial_fig.update_layout(
    title={
        'text': '🎬 Animated Semantic Landscape - Ideas Being Added',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 28, 'color': 'white'}
    },
    plot_bgcolor='#0E1117',
    paper_bgcolor='#0E1117',
    font=dict(color='white', size=12),
    xaxis=dict(
        title=f'{VIZ_METHOD.upper()} Component 1',
        showgrid=True,
        gridcolor='#333333',
        zeroline=False,
        range=[df['x'].min() - 1, df['x'].max() + 1]
    ),
    yaxis=dict(
        title=f'{VIZ_METHOD.upper()} Component 2',
        showgrid=True,
        gridcolor='#333333',
        zeroline=False,
        range=[df['y'].min() - 1, df['y'].max() + 1]
    ),
    height=700,
    hovermode='closest',
    legend=dict(
        bgcolor='rgba(30,30,30,0.8)',
        bordercolor='white',
        borderwidth=1,
        font=dict(size=11)
    ),
    updatemenus=[dict(
        type='buttons',
        showactive=False,
        buttons=[
            dict(label='▶ Play',
                 method='animate',
                 args=[None, dict(frame=dict(duration=100, redraw=True),
                                  fromcurrent=True,
                                  mode='immediate')]),
            dict(label='⏸ Pause',
                 method='animate',
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                   mode='immediate',
                                   transition=dict(duration=0))])
        ],
        x=0.1,
        y=1.15,
        xanchor='left',
        yanchor='top',
        bgcolor='#FF2F92',
        bordercolor='white',
        borderwidth=2,
        font=dict(color='white', size=14)
    )]
)

# Save animated HTML
animated_filename = f'animated_{OUTPUT_PREFIX}{len(df)}.html'
initial_fig.write_html(animated_filename)
print(f"   ✓ Animated visualization saved: {animated_filename}")

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("VISUALIZATION GENERATION COMPLETE!")
print("=" * 80)
print(f"\nGenerated files:")
print(f"  📊 Static (for report):     {static_filename}")
print(f"  🖱️  Interactive:             {interactive_filename}")
print(f"  🎬 Animated:                {animated_filename}")
print(f"\n  Total ideas visualized: {len(df)}")
print(f"  Method: {VIZ_METHOD.upper()}")
print(f"  Embedding dimensions: {embeddings.shape[1]}D → 2D")

if VIZ_METHOD == 'pca':
    print(f"  Variance explained: {sum(explained_var):.2%}")

print("\n✅ All visualizations ready for your exam report!")
print("=" * 80)
