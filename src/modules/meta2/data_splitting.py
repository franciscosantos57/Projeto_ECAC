"""
Data Splitting Strategies - Exercício 3
Implementa estratégias de split within-subject e between-subject.
"""

import numpy as np
from sklearn.model_selection import train_test_split


def split_within_subject(X, y, participant_ids, train_size=0.6, val_size=0.2, test_size=0.2, random_state=42):
    """
    Split within-subject: cada participante aparece nos 3 conjuntos (train/val/test).
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        participant_ids: Array de IDs dos participantes (n_samples,)
        train_size: Proporção para treino (default: 0.6)
        val_size: Proporção para validação (default: 0.2)
        test_size: Proporção para teste (default: 0.2)
        random_state: Seed para reprodutibilidade
        
    Returns:
        dict com keys: X_train, X_val, X_test, y_train, y_val, y_test,
                       participant_ids_train, participant_ids_val, participant_ids_test
    """
    # Valida proporções
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Proporções devem somar 1.0"
    
    # Identifica participantes únicos
    unique_participants = np.unique(participant_ids)
    
    # Inicializa listas para acumular splits
    X_train_list, X_val_list, X_test_list = [], [], []
    y_train_list, y_val_list, y_test_list = [], [], []
    participant_id_train_list, participant_id_val_list, participant_id_test_list = [], [], []
    
    # Para cada participante, faz split individual
    for participant_id in unique_participants:
        # Extrai dados deste participante
        mask = participant_ids == participant_id
        X_participant_id = X[mask]
        y_participant_id = y[mask]
        participant_id_array = participant_ids[mask]
        
        # Verifica se stratify é possível
        # Precisa de pelo menos 2 amostras por classe
        unique_classes, class_counts = np.unique(y_participant_id, return_counts=True)
        can_stratify = all(class_counts >= 2)
        
        # Split em train + temp (val + test)
        X_train_participant_id, X_temp, y_train_participant_id, y_temp, participant_id_train_participant_id, participant_id_temp = train_test_split(
            X_participant_id, y_participant_id, participant_id_array,
            test_size=(val_size + test_size),
            random_state=random_state,
            stratify=y_participant_id if can_stratify else None
        )
        
        # Split temp em val + test
        val_ratio = val_size / (val_size + test_size)
        unique_classes_temp, class_counts_temp = np.unique(y_temp, return_counts=True)
        can_stratify_temp = all(class_counts_temp >= 2)
        
        X_val_participant_id, X_test_participant_id, y_val_participant_id, y_test_participant_id, participant_id_val_participant_id, participant_id_test_participant_id = train_test_split(
            X_temp, y_temp, participant_id_temp,
            test_size=(1 - val_ratio),
            random_state=random_state,
            stratify=y_temp if can_stratify_temp else None
        )
        
        # Acumula
        X_train_list.append(X_train_participant_id)
        X_val_list.append(X_val_participant_id)
        X_test_list.append(X_test_participant_id)
        y_train_list.append(y_train_participant_id)
        y_val_list.append(y_val_participant_id)
        y_test_list.append(y_test_participant_id)
        participant_id_train_list.append(participant_id_train_participant_id)
        participant_id_val_list.append(participant_id_val_participant_id)
        participant_id_test_list.append(participant_id_test_participant_id)
    
    # Concatena todos os participantes
    return {
        'X_train': np.vstack(X_train_list),
        'X_val': np.vstack(X_val_list),
        'X_test': np.vstack(X_test_list),
        'y_train': np.hstack(y_train_list),
        'y_val': np.hstack(y_val_list),
        'y_test': np.hstack(y_test_list),
        'participant_ids_train': np.hstack(participant_id_train_list),
        'participant_ids_val': np.hstack(participant_id_val_list),
        'participant_ids_test': np.hstack(participant_id_test_list)
    }


def split_between_subject(X, y, participant_ids, train_size=9, val_size=3, test_size=3, random_state=42):
    """
    Split between-subject: participantes distintos em cada conjunto.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        participant_ids: Array de IDs dos participantes (n_samples,)
        train_size: Número de participantes para treino (default: 9)
        val_size: Número de participantes para validação (default: 3)
        test_size: Número de participantes para teste (default: 3)
        random_state: Seed para reprodutibilidade
        
    Returns:
        dict com keys: X_train, X_val, X_test, y_train, y_val, y_test,
                       participant_ids_train, participant_ids_val, participant_ids_test
    """
    # Identifica participantes únicos
    unique_participants = np.unique(participant_ids)
    n_participants = len(unique_participants)
    
    # Valida número de participantes
    assert train_size + val_size + test_size <= n_participants, \
        f"Requer {train_size + val_size + test_size} participantes, mas apenas {n_participants} disponíveis"
    
    # Shuffle participantes
    np.random.seed(random_state)
    shuffled_participants = np.random.permutation(unique_participants)
    
    # Divide participantes
    train_participants = shuffled_participants[:train_size]
    val_participants = shuffled_participants[train_size:train_size + val_size]
    test_participants = shuffled_participants[train_size + val_size:train_size + val_size + test_size]
    
    # Cria máscaras
    train_mask = np.isin(participant_ids, train_participants)
    val_mask = np.isin(participant_ids, val_participants)
    test_mask = np.isin(participant_ids, test_participants)
    
    return {
        'X_train': X[train_mask],
        'X_val': X[val_mask],
        'X_test': X[test_mask],
        'y_train': y[train_mask],
        'y_val': y[val_mask],
        'y_test': y[test_mask],
        'participant_ids_train': participant_ids[train_mask],
        'participant_ids_val': participant_ids[val_mask],
        'participant_ids_test': participant_ids[test_mask]
    }


def compare_splitting_strategies(within_split, between_split, verbose=True):
    """
    Compara estratégias within-subject vs between-subject.
    
    Args:
        within_split: Resultado de split_within_subject
        between_split: Resultado de split_between_subject
        verbose: Se True, imprime análise detalhada
        
    Returns:
        dict com estatísticas comparativas
    """
    if verbose:
        
        # Within-subject
        print("\n1. WITHIN-SUBJECT (dados do mesmo participante em todos os conjuntos)")
        print("-" * 60)
        print(f"  Train: {len(within_split['y_train'])} amostras, "
              f"{len(np.unique(within_split['participant_ids_train']))} participantes")
        print(f"  Val:   {len(within_split['y_val'])} amostras, "
              f"{len(np.unique(within_split['participant_ids_val']))} participantes")
        print(f"  Test:  {len(within_split['y_test'])} amostras, "
              f"{len(np.unique(within_split['participant_ids_test']))} participantes")
        
        # Between-subject
        print("\n2. BETWEEN-SUBJECT (participantes distintos em cada conjunto)")
        print("-" * 60)
        print(f"  Train: {len(between_split['y_train'])} amostras, "
              f"{len(np.unique(between_split['participant_ids_train']))} participantes")
        print(f"  Val:   {len(between_split['y_val'])} amostras, "
              f"{len(np.unique(between_split['participant_ids_val']))} participantes")
        print(f"  Test:  {len(between_split['y_test'])} amostras, "
              f"{len(np.unique(between_split['participant_ids_test']))} participantes")
        
        # Análise
        print("\n3. DISCUSSÃO")
        print("-" * 60)
        print("Within-Subject: mesmo participante nas três partições.")
        print("  • Prós: mais dados por conjunto; preserva variabilidade intra-individual.")
        print("  • Contras: pode introduzir data leakage e sobrestimar performance em sujeitos novos.")
        
        print("\nBetween-Subject: participantes distintos por partição.")
        print("  • Prós: avalia melhor a generalização para novos participantes; mais realista para deployment.")
        print("  • Contras: menos dados por conjunto e maior variância nas métricas entre splits.")
        
        print("\nConclusão: A melhor estratégia parece ser a abrodagem Between-Subject, uma vez que reflete")
        print("           melhor a capacidade de generalização para novos participantes, crucial em aplicações reais.")
        print("=" * 60)
    
    return {
        'within': {
            'train_samples': len(within_split['y_train']),
            'val_samples': len(within_split['y_val']),
            'test_samples': len(within_split['y_test']),
            'train_participants': len(np.unique(within_split['participant_ids_train'])),
            'val_participants': len(np.unique(within_split['participant_ids_val'])),
            'test_participants': len(np.unique(within_split['participant_ids_test']))
        },
        'between': {
            'train_samples': len(between_split['y_train']),
            'val_samples': len(between_split['y_val']),
            'test_samples': len(between_split['y_test']),
            'train_participants': len(np.unique(between_split['participant_ids_train'])),
            'val_participants': len(np.unique(between_split['participant_ids_val'])),
            'test_participants': len(np.unique(between_split['participant_ids_test']))
        }
    }

