"""
Dataset Scenarios - Exercício 3.4
Prepara três cenários diferentes de datasets após o split:
a) All features/embeddings
b) PCA-reduced (90% variância)
c) ReliefF-selected (top 15 features)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

# Importa funções existentes de PCA e ReliefF
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.modules.meta1.pca_analysis import apply_pca
from src.modules.meta1.relieff_selection import calculate_relieff_score


def prepare_all_scenarios(split_data, variance_threshold=0.90, top_k_features=15, verbose=True):
    """
    Prepara os três cenários (a, b, c) para um dataset já dividido.
    
    IMPORTANTE: PCA e ReliefF são calculados APENAS com dados de treino.
    
    Args:
        split_data: dict com X_train, X_val, X_test, y_train, y_val, y_test
        variance_threshold: variância acumulada para PCA (default: 0.90)
        top_k_features: número de features para ReliefF (default: 15)
        verbose: se True, imprime informações
        
    Returns:
        dict com três cenários: 'all', 'pca', 'relieff'
    """
    X_train = split_data['X_train']
    X_val = split_data['X_val']
    X_test = split_data['X_test']
    y_train = split_data['y_train']
    y_val = split_data['y_val']
    y_test = split_data['y_test']
    
    if verbose:
        print(f"\nPreparando cenários (train: {X_train.shape[0]} amostras, {X_train.shape[1]} features)...")
    
    # CENÁRIO A: All features (normalizado)
    scaler_all = StandardScaler()
    X_train_all = scaler_all.fit_transform(X_train)  # Fit apenas no train
    X_val_all = scaler_all.transform(X_val)
    X_test_all = scaler_all.transform(X_test)
    
    if verbose:
        print(f"  ✓ Cenário A (all): {X_train_all.shape[1]} features")
    
    # CENÁRIO B: PCA-reduced (90% variância)
    pca, X_train_pca = apply_pca(X_train_all, n_components=None)  # Fit no train normalizado
    
    # Encontra número de componentes para 90% variância
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_90 = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Aplica PCA com n_components_90
    pca_reduced, X_train_pca_reduced = apply_pca(X_train_all, n_components=n_components_90)
    X_val_pca_reduced = pca_reduced.transform(X_val_all)
    X_test_pca_reduced = pca_reduced.transform(X_test_all)
    
    if verbose:
        print(f"  ✓ Cenário B (PCA): {n_components_90} componentes ({cumulative_variance[n_components_90-1]*100:.1f}% variância)")
    
    # CENÁRIO C: ReliefF-selected (top 15 features)
    relieff_scores = calculate_relieff_score(
        X_train_all, y_train, 
        n_neighbors=10, 
        n_samples=min(100, len(X_train_all)),
        verbose=False
    )
    
    # Seleciona top-k features
    top_k = min(top_k_features, X_train_all.shape[1])
    top_indices = np.argsort(relieff_scores)[::-1][:top_k]
    
    X_train_relieff = X_train_all[:, top_indices]
    X_val_relieff = X_val_all[:, top_indices]
    X_test_relieff = X_test_all[:, top_indices]
    
    if verbose:
        print(f"  ✓ Cenário C (ReliefF): {top_k} features selecionadas")
    
    return {
        'all': {
            'X_train': X_train_all,
            'X_val': X_val_all,
            'X_test': X_test_all,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'scaler': scaler_all,
            'n_features': X_train_all.shape[1]
        },
        'pca': {
            'X_train': X_train_pca_reduced,
            'X_val': X_val_pca_reduced,
            'X_test': X_test_pca_reduced,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'pca': pca_reduced,
            'scaler': scaler_all,
            'n_components': n_components_90,
            'variance_explained': cumulative_variance[n_components_90-1]
        },
        'relieff': {
            'X_train': X_train_relieff,
            'X_val': X_val_relieff,
            'X_test': X_test_relieff,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'scaler': scaler_all,
            'selected_indices': top_indices,
            'relieff_scores': relieff_scores,
            'n_features': top_k
        }
    }
