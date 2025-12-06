"""
SMOTE Balancer - Balanceamento automático de dataset
"""

import numpy as np
from .data_augmentation import smote_generate_samples


def balance_dataset_smote(X_train, y_train, n_neighbors=5, verbose=True):
    """
    Balanceia dataset de treino usando SMOTE.
    Aumenta classes minoritárias até atingir o tamanho da classe majoritária.
    
    Args:
        X_train: Features de treino
        y_train: Labels de treino
        n_neighbors: Número de vizinhos para SMOTE
        verbose: Se True, imprime progresso
        
    Returns:
        X_balanced, y_balanced: Dataset balanceado
    """
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    max_count = class_counts.max()
    majority_class = unique_classes[class_counts.argmax()]
    
    if verbose:
        print(f"\n{'─' * 60}")
        print("BALANCEAMENTO COM SMOTE")
        print(f"{'─' * 60}")
        print(f"Classe majoritária: Atividade {int(majority_class)} com {max_count} amostras")
        print(f"Total original: {len(y_train)} amostras")
    
    X_balanced = [X_train]
    y_balanced = [y_train]
    
    for cls, count in zip(unique_classes, class_counts):
        if count < max_count:
            n_synthetic = max_count - count
            synthetic = smote_generate_samples(X_train, y_train, cls, n_synthetic, n_neighbors)
            X_balanced.append(synthetic)
            y_balanced.append(np.full(n_synthetic, cls))
            
            if verbose:
                print(f"  Atividade {int(cls)} (índice {int(cls)}): {count} → {max_count} (+{n_synthetic} sintéticas)")
    
    X_balanced = np.vstack(X_balanced)
    y_balanced = np.hstack(y_balanced)
    
    if verbose:
        print(f"Total balanceado: {len(y_balanced)} amostras")
        print(f"{'─' * 60}")
    
    return X_balanced, y_balanced
