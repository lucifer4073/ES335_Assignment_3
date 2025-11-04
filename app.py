import streamlit as st
import torch
import torch.nn.functional as F
import pickle
import os
import requests
import zipfile
from pathlib import Path

st.set_page_config(page_title="Next-Word Predictor", layout="wide")
st.title("Next-Word Prediction using MLP")

# AUTO-DOWNLOAD MODELS IF NOT PRESENT
@st.cache_resource
def ensure_models_exist():
    """Download models from release if not present locally"""
    if not os.path.exists('model_cat1_low_epochs.pt'):
        try:
            st.info("Downloading model weights (first run only)...")
            url = "https://github.com/lucifer4073/ES335_Assignment_3/releases/download/v1.0-q1/q1_models.zip"
            response = requests.get(url, timeout=60)
            with open('q1_models.zip', 'wb') as f:
                f.write(response.content)
            with zipfile.ZipFile('q1_models.zip', 'r') as z:
                z.extractall()
            os.remove('q1_models.zip')
            st.success("Models downloaded!")
        except Exception as e:
            st.error(f"Failed to download: {e}")

# Call this first
ensure_models_exist()

SEQUENCE_LENGTH = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_models_and_vocab():
    models = {'cat1': {}, 'cat2': {}}
    class MLPTextGenerator(torch.nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, sequence_length, num_hidden_layers, activation='relu'):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
            layers = []
            in_dim = embedding_dim * sequence_length
            for _ in range(num_hidden_layers):
                layers.append(torch.nn.Linear(in_dim, hidden_dim))
                if activation == 'relu':
                    layers.append(torch.nn.ReLU())
                else:
                    layers.append(torch.nn.Tanh())
                in_dim = hidden_dim
            layers.append(torch.nn.Linear(in_dim, vocab_size))
            self.network = torch.nn.Sequential(*layers)
        def forward(self, x):
            x = self.embedding(x)
            x = x.view(x.size(0), -1)
            return self.network(x)
    for var in ['low_epochs', 'medium_epochs', 'high_epochs']:
        try:
            ckpt = torch.load(f'model_cat1_{var}.pt', map_location=device)
            model = MLPTextGenerator(vocab_size=ckpt.get('vocab_size', 5000), embedding_dim=64, hidden_dim=1024, sequence_length=SEQUENCE_LENGTH, num_hidden_layers=2, activation='relu').to(device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            models['cat1'][var] = model
        except:
            pass
    for var in ['low_epochs', 'medium_epochs', 'high_epochs']:
        try:
            ckpt = torch.load(f'model_cat2_{var}.pt', map_location=device)
            model = MLPTextGenerator(vocab_size=ckpt.get('vocab_size', 8000), embedding_dim=64, hidden_dim=1024, sequence_length=SEQUENCE_LENGTH, num_hidden_layers=2, activation='relu').to(device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            models['cat2'][var] = model
        except:
            pass
    try:
        with open('word_index_cat1.pkl', 'rb') as f:
            word_index_cat1 = pickle.load(f)
        with open('index_word_cat1.pkl', 'rb') as f:
            index_word_cat1 = pickle.load(f)
    except:
        word_index_cat1, index_word_cat1 = {}, {}
    try:
        with open('word_index_cat2.pkl', 'rb') as f:
            word_index_cat2 = pickle.load(f)
        with open('index_word_cat2.pkl', 'rb') as f:
            index_word_cat2 = pickle.load(f)
    except:
        word_index_cat2, index_word_cat2 = {}, {}
    return models, (word_index_cat1, index_word_cat1, word_index_cat2, index_word_cat2)

def generate_text(model, seed_text, word_index, index_word, num_words=5, temperature=1.0):
    model.eval()
    words = seed_text.lower().split()
    generated = words.copy()
    with torch.no_grad():
        for _ in range(num_words):
            context = generated[-SEQUENCE_LENGTH:] if len(generated) >= SEQUENCE_LENGTH else generated
            if len(context) < SEQUENCE_LENGTH:
                context = ['<PAD>'] * (SEQUENCE_LENGTH - len(context)) + context
            indices = [word_index.get(w, 0) for w in context]
            x = torch.tensor([indices], dtype=torch.long).to(device)
            logits = model(x)
            probs = F.softmax(logits / max(temperature, 0.1), dim=-1)
            next_idx = torch.multinomial(probs[0], 1).item()
            if next_idx < len(index_word):
                generated.append(index_word[next_idx])
    return ' '.join(generated)

models, vocabs = load_models_and_vocab()
word_index_cat1, index_word_cat1, word_index_cat2, index_word_cat2 = vocabs
col1, col2 = st.columns([1, 3])
with col1:
    st.subheader("Configuration")
    dataset = st.radio("Dataset", ["Sherlock Holmes", "Linux Kernel"])
    model_variant = st.radio("Model", ["Low Epochs", "Medium Epochs", "High Epochs"])
    num_gen = st.slider("Words to generate", 1, 20, 5)
    temp = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
    cat = 'cat1' if 'Sherlock' in dataset else 'cat2'
    var = model_variant.lower().replace(' ', '_')
    word_idx = word_index_cat1 if cat == 'cat1' else word_index_cat2
    idx_word = index_word_cat1 if cat == 'cat1' else index_word_cat2
with col2:
    st.subheader("Generation")
    seed = st.text_input("Seed text:", value="the adventure" if cat == 'cat1' else "int main")
    if st.button("Generate", use_container_width=True):
        if seed and var in models.get(cat, {}):
            try:
                text = generate_text(models[cat][var], seed, word_idx, idx_word, num_gen, temp)
                st.success("SUCCESS - Generated!")
                st.info(f"Generated: {text}")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Length", len(text.split()))
                col_b.metric("Temperature", f"{temp:.1f}")
                col_c.metric("Dataset", "Cat I" if cat == 'cat1' else "Cat II")
            except Exception as e:
                st.error(f"ERROR: {str(e)[:100]}")
st.sidebar.info("Model Variants:\n- Low: Underfitting\n- Medium: Balanced\n- High: Overfitting")