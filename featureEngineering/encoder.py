from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import json
import os
import logging
from typing import Dict, List, Optional

def embed_posts(user_data: Dict,
                     model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                     batch_size: int = 100,
                     device: Optional[str] = None,
                     output_path: Optional[str] = None) -> List[Dict]:
    """
    Encode user post-like data using SentenceTransformer in batches.
    
    Args:
        user_data: Dict with keys like 'posts', 'reposts', 'replies', etc., each a list of dicts with 'text'
        model_name: SentenceTransformer model to use
        batch_size: Number of texts to embed per batch
        device: Force specific device (e.g. 'cuda', 'cpu', 'mps'); auto-detect if None
        output_path: Optional file path to save result as JSON
    
    Returns:
        List of dicts with 'text' and 'embedding'
    """
    print(f"Encoding user texts using model: {model_name}")
    
    # Gather all valid text entries across all sources
    documents = []
    for key in ['posts', 'reposts', 'replies', 'likes']:
        for item in user_data.get(key, []):
            text = item.get('text', '')
            if isinstance(text, str) and text.strip():
                documents.append(text.strip())

    if not documents:
        print("No valid text found for embedding.")
        return []
    
    print(f"Collected {len(documents)} texts for embedding.")

    # Load model
    model = SentenceTransformer(model_name)

    # Set device
    if device:
        model = model.to(device)
        print(f"Using specified device: {device}")
    else:
        if torch.cuda.is_available():
            model = model.to('cuda')
            print("Using CUDA")
        elif torch.backends.mps.is_available():
            model = model.to('mps')
            print("Using MPS")
        else:
            print("Using CPU")

    # Batch processing
    all_embeddings = []
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} of {(len(documents) + batch_size - 1)//batch_size}")
        embeddings = model.encode(batch, convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=True)
        all_embeddings.append(embeddings.cpu().numpy())

    all_embeddings = np.vstack(all_embeddings)
    result = [{'text': text, 'embedding': emb.tolist()} for text, emb in zip(documents, all_embeddings)]

    print(f"Successfully encoded {len(result)} documents.")

    # Save if path is provided
    return result
