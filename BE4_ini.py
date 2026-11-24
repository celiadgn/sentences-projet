# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 10:50:37 2026

@author: dejea
"""

import json
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer

import warnings
warnings.filterwarnings('ignore')

# For embeddings and similarity computation
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    print("Required libraries imported successfully!")
except ImportError as e:
    print(f"Missing library: {e}")
    print("Please install with: pip install sentence-transformers scikit-learn networkx")

np.random.seed(42)

from scipy.sparse import find, csr_matrix
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import norm
from IPython.core.display import HTML

# des options permettent de limiter (ou non) le nombre de lignes/colonnes affichées
# par exemple :
# pd.set_option('display.max_rows', None)

# cette fonction permet d'afficher une "jolie" représentation du vecteur v
# ARGS :
#   v : le vecteur à afficher (par ex. une ligne de la matrice X)
#   features : le vocabulaire
#   top_n : le nombre de mots maximum à afficher
def print_feats(v, features, top_n = 30):
    _, ids, values = find(v)
    feats = [(ids[i], values[i], features[ids[i]]) for i in range(len(list(ids)))]
    top_feats = sorted(feats, key=lambda x: x[1], reverse=True)[0:top_n]
    return pd.DataFrame({"word" : [t[2] for t in top_feats], "value": [t[1] for t in top_feats]})   

# fonction qui permet d'afficher plusieurs tables pandas côte à côte (c'est cadeau)
def display_side_by_side(dfs:list, captions:list):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    output = ""
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += df.style.set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
        output += "&emsp;"
        #output += "\xa0\xa0\xa0"
    display(HTML(output))
    
    
    

def load_corpus(file_path: str) -> Dict[str, Dict]:
    """
    Load corpus data from JSONL file.
    Returns
    """
    corpus = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc['_id']] = doc
    return corpus


def load_queries(file_path: str) -> Dict[str, Dict]:
    """
    Load query data from JSONL file.
    Returns dictionary mapping query IDs to query data.
    """
    queries = {}
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            queries[q['_id']] = q
    return queries    


def load_qrels(file_path: str) -> Dict[str, Dict[str, int]]:
    """
    Load relevance judgments from TSV file.
    Returns dictionary mapping query IDs to candidate relevance scores.
    """
    qrels = defaultdict(dict)
    with open(file_path, encoding='utf-8') as f:
        next(f)  # saute l'en-tête "query-id corpus-id score"
        for line in f:
            q_id, doc_id, score = line.split()
            qrels[q_id][doc_id] = int(score)
    return qrels

# Load the dataset
print("Loading dataset...")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_BE2')

corpus = load_corpus(os.path.join(DATA_DIR, 'corpus.jsonl'))
queries = load_queries(os.path.join(DATA_DIR, 'queries.jsonl'))
qrels_valid = load_qrels(os.path.join(DATA_DIR, 'valid.tsv'))



print(f"Loaded {len(corpus)} documents in corpus")
print(f"Loaded {len(queries)} queries")
print(f"Loaded relevance for {len(qrels_valid)} queries (dataset)")


# Partie 2 essai 1 Colombe
# ---------- Statistiques simples ----------
# Nombre total de paires requête / document
nb_paires = sum(len(cands) for cands in qrels_valid.values())
print(f"Nombre de paires requête / document : {nb_paires}")

# Proportion de documents pertinents par requête
prop_pos = [sum(cands.values()) / len(cands) for cands in qrels_valid.values()]
nb_cands = [len(cands) for cands in qrels_valid.values()]
print(f"Candidats par requête : moyenne {np.mean(nb_cands):.1f} (min {min(nb_cands)}, max {max(nb_cands)})")
print(f"Proportion de pertinents par requête : moyenne {np.mean(prop_pos):.2%} "
      f"(min {min(prop_pos):.2%}, max {max(prop_pos):.2%})")
print(f"Proportion de non pertinents par requête : moyenne {1 - np.mean(prop_pos):.2%}")

q_id = next(iter(qrels_valid))
print("Champs d'une requête :", queries[q_id].keys())
print("Champs d'un document :", corpus[next(iter(corpus))].keys())
print("\nRequête :", queries[q_id]['text'])

positifs = [d for d, s in qrels_valid[q_id].items() if s == 1]
negatifs = [d for d, s in qrels_valid[q_id].items() if s == 0]

print("Candidats POSITIFS :")
for d in positifs[:3]:
    print("  +", corpus[d]['title'])

print("Candidats NÉGATIFS :")
for d in negatifs[:3]:
    print("  -", corpus[d]['title'])