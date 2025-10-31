"""
Análise e visualização do feature set extraído.
Inclui estatísticas, visualizações e exportação de dados.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from src.utils.constants import ACTIVITY_NAMES, DEVICE_NAMES


def analyze_feature_set(feature_matrix, labels, metadata, feature_names):
    """
    Analisa o feature set extraído e gera estatísticas.
    
    Args:
        feature_matrix: Array [n_windows, n_features]
        labels: Array [n_windows] com IDs das atividades
        metadata: Lista com metadados de cada janela
        feature_names: Lista com nomes das features
    """
    n_windows, n_features = feature_matrix.shape
    
    print(f"\nFeature set: {n_windows} janelas × {n_features} features")
    
    # Verifica qualidade
    n_nan = np.sum(np.isnan(feature_matrix))
    n_inf = np.sum(np.isinf(feature_matrix))
    if n_nan > 0 or n_inf > 0:
        print(f"ATENÇÃO: {n_nan} NaN, {n_inf} Inf detectados!")
    else:
        print("Dataset limpo")


def create_feature_visualizations(feature_matrix, labels, metadata, feature_names, 
                                  output_dir="plots/exercicio_4.2_features"):
    """
    Cria visualizações do feature set extraído.
    
    NOTA: Visualizações desativadas - exercício 4 não requer plots (exceto 4.3 PCA).
    """
    print("Visualizações desativadas - apenas análise numérica ativa.")
    # Gráficos removidos conforme especificação do projeto
    pass


def save_feature_set(feature_matrix, labels, metadata, feature_names, output_dir):
    """
    Salva o feature set em formato CSV e NumPy.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Salva em formato NumPy (.npz)
    npz_path = os.path.join(output_dir, 'feature_set.npz')
    np.savez(npz_path,
             features=feature_matrix,
             labels=labels,
             feature_names=feature_names)
    
    # Salva informações em formato texto
    txt_path = os.path.join(output_dir, 'feature_info.txt')
    with open(txt_path, 'w') as f:
        f.write("FEATURE SET INFORMATION\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dimensions: {feature_matrix.shape[0]} windows × {feature_matrix.shape[1]} features\n\n")
        f.write("Feature Names:\n")
        for i, name in enumerate(feature_names, 1):
            f.write(f"  {i:3d}. {name}\n")
    
    print(f"Feature set salvo: {npz_path}")
