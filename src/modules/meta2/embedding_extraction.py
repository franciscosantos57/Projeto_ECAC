"""
Módulo para análise de Embedding Features usando Transfer Learning
Meta 2 - Exercício 2.1

Os embeddings são extraídos durante a extração de features na Meta 1 (Exercício 4.2),
garantindo perfeito alinhamento entre Features Dataset e Embeddings Dataset.

Este módulo fornece funções para:
- Carregar embeddings já extraídos
- Analisar e comparar embeddings com features handcrafted

Referências:
- Projeto harnet5: https://github.com/OxWearables/ssl-wearables
- Paper: https://www.nature.com/articles/s41746-024-01062-3
"""
import os
import numpy as np


def load_embeddings_dataset(embeddings_path: str = "data/features/embeddings_set.npz"):
    """
    Carrega dataset de embeddings já extraído.
    
    Args:
        embeddings_path: Caminho do ficheiro .npz com embeddings
        
    Returns:
        tuple: (embeddings, labels, participant_ids, devices)
            - embeddings: Array [n_segments, 512] com embeddings
            - labels: Array [n_segments] com IDs de atividades
            - participant_ids: Array [n_segments] com IDs dos participantes
            - devices: Array [n_segments] com IDs dos dispositivos
    """
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"Embeddings não encontrados: {embeddings_path}\n"
            f"Execute primeiro o Exercício 4.2 com extract_embeddings=True"
        )
    
    data = np.load(embeddings_path, allow_pickle=True)
    
    embeddings = data['embeddings']
    labels = data['labels']
    participant_ids = data['participant_ids']
    devices = data['devices']
    
    return embeddings, labels, participant_ids, devices
