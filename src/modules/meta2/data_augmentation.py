"""
Módulo para Data Augmentation usando SMOTE
Meta 2 - Exercício 1
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import os


def analyze_dataset_balance(X: np.ndarray, y: np.ndarray, verbose: bool = True) -> dict:
    """
    Analisa o balanço do dataset entre as diferentes atividades.
    
    Exercício 1.1: Verifica se o dataset está balanceado.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        verbose: Se True, imprime informações detalhadas
        
    Returns:
        dict: Dicionário com estatísticas de balanço
    """
    unique_activities = np.unique(y)
    activity_counts = {}
    
    for activity in unique_activities:
        count = np.sum(y == activity)
        activity_counts[activity] = count
    
    total_samples = len(y)
    max_count = max(activity_counts.values())
    min_count = min(activity_counts.values())
    
    # Calcula o rácio de desbalanço
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    # Considera balanceado se o rácio for menor que 1.5
    is_balanced = imbalance_ratio < 1.5
    
    results = {
        'activity_counts': activity_counts,
        'total_samples': total_samples,
        'max_count': max_count,
        'min_count': min_count,
        'imbalance_ratio': imbalance_ratio,
        'is_balanced': is_balanced
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("ANÁLISE DE BALANÇO DO DATASET")
        print("=" * 60)
        print(f"\nTotal de amostras: {total_samples}")
        print(f"Número de atividades: {len(unique_activities)}")
        print("\nDistribuição por atividade:")
        print("-" * 60)
        
        for activity in sorted(activity_counts.keys()):
            count = activity_counts[activity]
            percentage = (count / total_samples) * 100
            print(f"Atividade {int(activity)}: {count:5d} amostras ({percentage:5.2f}%)")
        
        print("-" * 60)
        print(f"\nMáximo de amostras: {max_count}")
        print(f"Mínimo de amostras: {min_count}")
        print(f"Rácio de desbalanço: {imbalance_ratio:.2f}")
        print(f"\nDataset está balanceado? {'SIM' if is_balanced else 'NÃO'}")
        
        if not is_balanced:
            print(f"   A atividade com mais amostras tem {imbalance_ratio:.2f}x mais dados")
            print(f"   que a atividade com menos amostras.")
        
        print("=" * 60)
    
    return results


def smote_generate_samples(X: np.ndarray, y: np.ndarray, target_activity: int, 
                           k_samples: int, n_neighbors: int = 5) -> np.ndarray:
    """
    Implementa o algoritmo SMOTE para gerar amostras sintéticas.
    
    Exercício 1.2: Gera K novas amostras para uma atividade específica A.
    
    SMOTE (Synthetic Minority Over-sampling Technique):
    Para cada amostra da classe minoritária:
    1. Encontra os k vizinhos mais próximos da mesma classe
    2. Seleciona aleatoriamente um dos vizinhos
    3. Gera uma amostra sintética interpolando entre a amostra original e o vizinho
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        target_activity: Atividade para a qual gerar amostras sintéticas
        k_samples: Número de amostras sintéticas a gerar
        n_neighbors: Número de vizinhos a considerar no SMOTE
        
    Returns:
        np.ndarray: Matriz com as k_samples amostras sintéticas geradas (k_samples, n_features)
    """
    # Filtra amostras da atividade alvo
    activity_mask = (y == target_activity)
    X_activity = X[activity_mask]
    
    if len(X_activity) == 0:
        raise ValueError(f"Nenhuma amostra encontrada para a atividade {target_activity}")
    
    if len(X_activity) < n_neighbors:
        n_neighbors = len(X_activity) - 1
        if n_neighbors < 1:
            raise ValueError(f"Amostras insuficientes para SMOTE (mínimo 2 necessárias)")
    
    synthetic_samples = []
    
    for _ in range(k_samples):
        # Seleciona uma amostra aleatória da atividade
        idx = np.random.randint(0, len(X_activity))
        sample = X_activity[idx]
        
        # Calcula distâncias euclidianas para todas as outras amostras
        distances = np.linalg.norm(X_activity - sample, axis=1)
        
        # Encontra os k vizinhos mais próximos (excluindo a própria amostra)
        nearest_indices = np.argsort(distances)[1:n_neighbors+1]
        
        # Seleciona aleatoriamente um dos vizinhos
        neighbor_idx = np.random.choice(nearest_indices)
        neighbor = X_activity[neighbor_idx]
        
        # Gera amostra sintética por interpolação linear
        # new_sample = sample + λ * (neighbor - sample), onde λ ∈ [0, 1]
        lambda_val = np.random.random()
        synthetic_sample = sample + lambda_val * (neighbor - sample)
        
        synthetic_samples.append(synthetic_sample)
    
    return np.array(synthetic_samples)


def visualize_smote_samples(X: np.ndarray, y: np.ndarray, participant_id: int,
                           target_activity: int, synthetic_samples: np.ndarray,
                           output_dir: str = "plots/meta2/exercicio_1.3_smote",
                           feature_names: Optional[list] = None) -> str:
    """
    Visualiza amostras sintéticas geradas pelo SMOTE em scatter plot 2D.
    Usa acc_x_std (eixo X) e gyro_x_std (eixo Y) para melhor separação visual.
    
    Args:
        X: Feature matrix do participante
        y: Labels do participante
        participant_id: ID do participante
        target_activity: Atividade alvo
        synthetic_samples: Amostras sintéticas geradas
        output_dir: Diretório para salvar o plot
        feature_names: Lista com nomes das features
        
    Returns:
        str: Caminho do arquivo salvo
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Índices das features: acc_x_std e gyro_x_std (melhor separação entre atividades)
    feature_idx_1 = feature_names.index('acc_x_std') if feature_names else 2
    feature_idx_2 = feature_names.index('gyro_x_std') if feature_names else 56
    
    # Configura o plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define cores para cada atividade
    unique_activities = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_activities)))
    color_map = {act: colors[i] for i, act in enumerate(unique_activities)}
    
    # Plota amostras originais
    for activity in unique_activities:
        mask = (y == activity)
        X_activity = X[mask]
        
        if activity == target_activity:
            ax.scatter(X_activity[:, feature_idx_1], X_activity[:, feature_idx_2], 
                      c=[color_map[activity]], alpha=0.6, s=50,
                      label=f'Atividade {int(activity)} (original)',
                      edgecolors='black', linewidths=0.5)
        else:
            ax.scatter(X_activity[:, feature_idx_1], X_activity[:, feature_idx_2], 
                      c=[color_map[activity]], alpha=0.3, s=30,
                      label=f'Atividade {int(activity)}',
                      edgecolors='none')
    
    # Plota amostras sintéticas
    ax.scatter(synthetic_samples[:, feature_idx_1], synthetic_samples[:, feature_idx_2],
              c='red', marker='*', s=400, alpha=0.9,
              label=f'Sintéticas (Atividade {int(target_activity)})',
              edgecolors='darkred', linewidths=2)
    
    # Configurações do gráfico
    xlabel = feature_names[feature_idx_1] if feature_names else 'acc_x_std'
    ylabel = feature_names[feature_idx_2] if feature_names else 'gyro_x_std'
    
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(f'SMOTE - Amostras Sintéticas para Atividade {int(target_activity)}\n' +
                f'Participante {participant_id} | {len(synthetic_samples)} amostras geradas',
                fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(loc='best', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Salva o plot
    filename = f"smote_participant{participant_id}_activity{int(target_activity)}.png"
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath


def demonstrate_smote(X: np.ndarray, y: np.ndarray, participant_id: int,
                     target_activity: int, k_samples: int = 3,
                     n_neighbors: int = 5, output_dir: str = "plots/meta2/exercicio_1.3_smote",
                     verbose: bool = True, feature_names: Optional[list] = None) -> dict:
    """
    Gera k amostras sintéticas usando SMOTE e visualiza em scatter plot 2D.
    
    Args:
        X: Feature matrix do participante
        y: Labels do participante
        participant_id: ID do participante
        target_activity: Atividade para gerar amostras sintéticas
        k_samples: Número de amostras sintéticas a gerar
        n_neighbors: Número de vizinhos para SMOTE
        output_dir: Diretório para salvar visualizações
        verbose: Se True, imprime informações
        feature_names: Lista com nomes das features
        
    Returns:
        dict: Resultados da demonstração
    """
    if verbose:
        print("\n" + "=" * 60)
        print(f"DEMONSTRAÇÃO SMOTE - PARTICIPANTE {participant_id}")
        print("=" * 60)
        print(f"Atividade alvo: {int(target_activity)}")
        print(f"Amostras a gerar: {k_samples}")
        print(f"Vizinhos (k): {n_neighbors}")
        print("-" * 60)
    
    original_count = np.sum(y == target_activity)
    
    if verbose:
        print(f"Amostras originais da atividade {int(target_activity)}: {original_count}")
    
    # Gera amostras sintéticas
    if verbose:
        print(f"\nGerando {k_samples} amostras sintéticas usando SMOTE...")
    
    synthetic_samples = smote_generate_samples(X, y, target_activity, k_samples, n_neighbors)
    
    if verbose:
        print(f"✓ Amostras sintéticas geradas: {synthetic_samples.shape}")
        print(f"  Dimensões: {synthetic_samples.shape[0]} amostras × {synthetic_samples.shape[1]} features")
    
    # Visualiza resultados
    if verbose:
        print(f"\nCriando visualização 2D...")
    
    plot_path = visualize_smote_samples(X, y, participant_id, target_activity, 
                                       synthetic_samples, output_dir, feature_names)
    
    if verbose:
        print(f"✓ Gráfico salvo: {plot_path}")
        print("=" * 60)
        print("RESUMO DAS AMOSTRAS SINTÉTICAS")
        print("=" * 60)
        
        idx1 = feature_names.index('acc_x_std') if feature_names else 2
        idx2 = feature_names.index('gyro_x_std') if feature_names else 56
        
        print(f"\nFeatures visualizadas (acc_x_std e gyro_x_std):")
        print("-" * 60)
        for i, sample in enumerate(synthetic_samples, 1):
            print(f"Amostra {i}: acc_x_std={sample[idx1]:8.4f}, gyro_x_std={sample[idx2]:8.4f}")
        print("=" * 60)
    
    return {
        'synthetic_samples': synthetic_samples,
        'original_count': original_count,
        'plot_path': plot_path,
        'participant_id': participant_id,
        'target_activity': target_activity
    }
