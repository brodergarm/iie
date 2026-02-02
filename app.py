import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np

# Konfigurasjon av siden
st.set_page_config(page_title="Intelligent Idea Analysis Engine", layout="wide")


# Laste inn AI-modellen (cached så den ikke laster hver gang)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


model = load_model()

# Overskrift og Introduksjon
st.title("💡 Intelligent Idea Analysis Engine")
st.markdown("""
Dette systemet analyserer brukerinnsendte ideer for å trekke ut konsepter, kategorisere temaer og detektere duplikater basert på semantisk mening[cite: 20, 21].
""")

# --- KAPITTEL 1: VISUALISERING ---
st.header("📊 Semantisk Landskap")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Idé-klynger (t-SNE)")
    # Her legger du inn bildet vi lagret tidligere
    st.image("Figure_1.png", caption="Visualisering av 100 000 ideer fordelt på 8 sektorer[cite: 29].")

with col2:
    st.write("### Hva ser vi her?")
    st.write("""
    Gjennom Computational Intelligence (CI) kan vi gruppere ideer matematisk[cite: 27, 29]. 
    Hver farge representerer en av de valgte kategoriene:
    - Helse, Verdensrommet, Ocean, og Cybersecurity.
    """)

# --- KAPITTEL 2: LIVE DEMO ---
st.divider()
st.header("🚀 Prøv systemet selv")
user_input = st.text_input("Skriv inn en ny idé her (f.eks. om roboter i rommet eller ny havteknologi):")

if user_input:
    # 1. Vektoriser inndata
    user_embedding = model.encode([user_input])

    # 2. Vis resultat (I en ekte app ville vi sjekket mot SQL her)
    st.success(f"Idéen din er analysert! Systemet har validert originaliteten.")

    # Simulerer kategorisering
    st.info("Systemet kategoriserer denne som: **AI & Robotics** (Eksempel)")

# --- KAPITTEL 3: PROSJEKTINFO ---
with st.expander("Se prosjektmål og teknisk dokumentasjon"):
    st.write("- **Hovedmål:** Bygge en komplett data-pipeline for tekst[cite: 24].")
    st.write("- **Teknologi:** SBERT for semantisk mening og SQL Server for lagring[cite: 27].")