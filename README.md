# Intelligent Idea Analysis Engine

**Exam Project** - Christian Garmann Schjelderup  

---

## Project Overview

This project demonstrates Computational Intelligence through an intelligent system that:

- Understands semantic meaning in text (not just word matching)
- Detects duplicates based on conceptual similarity
- Validates new ideas against a database of 1600+ existing ideas
- Visualizes the semantic landscape in 2D

---

## Demonstration of Computational Intelligence

### 1. Semantic Understanding

The system recognizes these as the same idea:
- "App for sharing tools with neighbors"
- "Platform for neighborhood equipment lending"

The model uses deep learning with a transformer architecture to understand meaning rather than relying on simple word matching.

### 2. AI Model

| Component | Details |
|-----------|---------|
| **Model** | all-MiniLM-L6-v2 |
| **Architecture** | BERT-based Sentence Transformer |
| **Dimensions** | 384-dimensional vector representation |

### 3. Vector-Based Comparison
```
Idea 1: "Tool sharing app"                → [0.23, -0.45, 0.12, ...]  (384 dimensions)
Idea 2: "Equipment lending platform"      → [0.21, -0.43, 0.15, ...]  (384 dimensions)

Cosine Similarity = 0.89  (High similarity indicates same concept)
```

### 4. Automatic Clustering

- t-SNE/PCA algorithms group ideas automatically without manual labeling
- Similar ideas cluster together visually
- This represents unsupervised learning where the AI identifies patterns independently

---

## Running the Project

### 1. Streamlit Web Application (Primary Interface)
```bash
streamlit run app.py
```

**Features:**
- Input validation for new ideas
- Visualization of idea positioning in semantic space
- Display of top 5 most similar existing ideas
- Interactive mapping with hover functionality

**Live Demo**: https://intelengine.streamlit.app/

### 2. Visualization Generation
```bash
python visualization.py
```

**Output Files:**
- Static PNG image for report
- Interactive HTML visualization

### 3. Baseline Model
```bash
python dm_baseline.py
```

**Output:**
- TF-IDF + K-Means clustering analysis
- Elbow and Silhouette plots

---

## File Structure
```
project/
│
├── app.py                       # Main Streamlit application
├── visualization.py             # Visualization generation script
├── dm_baseline.py              # Baseline TF-IDF + K-Means model
├── idea_generator.py           # Data generation script
├── Embedding_engine.py         # BERT embedding generation
├── idea_sample.csv             # Dataset (1600 ideas)
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

---

## Technical Implementation

### Processing Pipeline
```
1. Input: "App for dog walking with GPS"
2. Preprocessing: Tokenization
3. Embedding: 384D vector representation via BERT
4. Comparison: Cosine similarity calculation against database
5. Decision Logic: 
   - > 0.85 → REJECT (duplicate detected)
   - > 0.65 → WARNING (semantically similar)
   - ≤ 0.65 → APPROVE (sufficiently unique)
6. Visualization: Semantic map update
```

### Similarity Calculation
```python
similarity = 1 - cosine_distance

cosine_distance = 1 - (A · B) / (||A|| * ||B||)

where:
A = vector representation of idea 1
B = vector representation of idea 2
```

### Dimensionality Reduction

Transformation from 384D to 2D using:
- **PCA** (Principal Component Analysis) - Fast, linear transformation
- **t-SNE** (t-Distributed Stochastic Neighbor Embedding) - Better clustering visualization

---

## Comparison: Data Mining vs Computational Intelligence

| Aspect | Traditional Data Mining | Computational Intelligence |
|--------|------------------------|----------------------------|
| **Methodology** | TF-IDF + Keyword matching | Deep Learning Transformer |
| **Understanding** | Surface-level (lexical) | Deep semantic (contextual) |
| **Duplicate Detection** | "dog" = "dog" (matches)<br>"dog" ≠ "canine" (fails) | "dog" = "canine" (succeeds) |

---

## Performance Metrics

### Validation Performance

- **Processing speed**: ~0.2 seconds per idea
- **Database size**: 1600 ideas
- **Duplicate detection**: High accuracy on test cases

### Clustering Performance

- **Silhouette Score (TF-IDF)**: 0.1893 (weak clustering)
- **Silhouette Score (BERT)**: Not directly comparable (supervised threshold-based)

---

## Key Achievements

This project successfully demonstrates:

- Complete data processing pipeline
- Baseline comparison with traditional Data Mining (TF-IDF + K-Means)
- Advanced Computational Intelligence implementation (BERT)
- Semantic comprehension (recognizes conceptual similarity)
- Real-time validation system
- Multiple visualization formats
- Deployed web application

---

## References

- Reimers & Gurevych (2019) - "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- Vaswani et al. (2017) - "Attention Is All You Need"
- van der Maaten & Hinton (2008) - "Visualizing Data using t-SNE"

---

**Project by Christian Garmann Schjelderup**
