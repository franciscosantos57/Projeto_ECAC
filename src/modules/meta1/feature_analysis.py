"""
Análise e visualização do feature set extraído.
Inclui estatísticas, visualizações e exportação de dados.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import os


def load_feature_set(output_dir="data/features"):
    """
    Carrega feature set e embeddings de ficheiros .npz salvos anteriormente.
    
    Args:
        output_dir: Diretório onde os ficheiros estão guardados
        
    Returns:
        tuple: (feature_matrix, labels, metadata, feature_names, embeddings)
               ou None se ficheiros não existirem
    """
    feature_path = os.path.join(output_dir, 'feature_set.npz')
    embeddings_path = os.path.join(output_dir, 'embeddings_set.npz')
    
    if not os.path.exists(feature_path):
        return None
    
    # Carrega features
    data = np.load(feature_path, allow_pickle=True)
    feature_matrix = data['features']
    labels = data['labels']
    participant_ids = data['participant_ids']
    devices = data['devices']
    feature_names = data['feature_names'].tolist()
    
    # Reconstrói metadata
    metadata = []
    for i in range(len(labels)):
        metadata.append({
            'participant_id': int(participant_ids[i]),
            'device': int(devices[i]),
            'activity': int(labels[i])
        })
    
    # Carrega embeddings se existirem
    embeddings = None
    if os.path.exists(embeddings_path):
        emb_data = np.load(embeddings_path, allow_pickle=True)
        embeddings = emb_data['embeddings']
    
    return feature_matrix, labels, metadata, feature_names, embeddings


def analyze_feature_set(feature_matrix, labels, metadata, feature_names):
    """
    Analisa o feature set extraído e gera estatísticas.
    
    Args:
        feature_matrix: Array [n_windows, n_features]
        labels: Array [n_windows] com IDs das atividades
        metadata: Lista com metadados de cada janela
        feature_names: Lista com nomes das features
    """
    # Verifica qualidade (silenciosamente)
    n_nan = np.sum(np.isnan(feature_matrix))
    n_inf = np.sum(np.isinf(feature_matrix))
    if n_nan > 0 or n_inf > 0:
        print(f"ATENÇÃO: {n_nan} NaN, {n_inf} Inf detectados no feature set!")


def save_feature_set(feature_matrix, labels, metadata, feature_names, output_dir, embeddings=None):
    """
    Guarda feature set e embeddings (se disponíveis) em formato NumPy.
    
    Args:
        feature_matrix: Array [n_windows, 66] com features handcrafted
        labels: Array [n_windows] com IDs das atividades
        metadata: Lista de dicts com info de cada janela
        feature_names: Lista com nomes das features
        output_dir: Diretório de saída
        embeddings: Array [n_windows, 512] com embeddings ou None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extrai arrays de metadata para ambos os ficheiros
    participant_ids = np.array([m['participant_id'] for m in metadata], dtype=np.int32)
    devices = np.array([m['device'] for m in metadata], dtype=np.int32)
    
    # Guarda features handcrafted (ESTRUTURA CONSISTENTE)
    npz_path = os.path.join(output_dir, 'feature_set.npz')
    np.savez(npz_path,
             features=feature_matrix,
             labels=labels,
             participant_ids=participant_ids,
             devices=devices,
             feature_names=feature_names)
    
    # Guarda embeddings silenciosamente se disponíveis (ESTRUTURA CONSISTENTE)
    if embeddings is not None:
        embeddings_path = os.path.join(output_dir, 'embeddings_set.npz')
        np.savez_compressed(
            embeddings_path,
            embeddings=embeddings,
            labels=labels,
            participant_ids=participant_ids,
            devices=devices
        )
    
    # Guarda informações em formato texto (silenciosamente)
    txt_path = os.path.join(output_dir, 'feature_info.txt')
    with open(txt_path, 'w') as f:
        f.write("FEATURE SET INFORMATION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dimensions: {feature_matrix.shape[0]} windows × {feature_matrix.shape[1]} features\n\n")
        f.write("Feature Names:\n")
        for i, name in enumerate(feature_names, 1):
            f.write(f"  {i:3d}. {name}\n")
