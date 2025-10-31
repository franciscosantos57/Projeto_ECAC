"""
EXERCÍCIO 4.5: Implementação do ReliefF
ReliefF é um algoritmo de seleção de features baseado em instâncias que avalia 
a relevância de cada feature baseando-se na diferença entre instâncias próximas 
da mesma classe (near-hits) e de classes diferentes (near-misses).

ALGORITMO:
1. Para cada instância aleatória:
   - Encontra k nearest hits (mesma classe)
   - Encontra k nearest misses (classe diferente)
   - Atualiza pesos baseado nas diferenças
   
2. Peso atualizado:
   W[f] = W[f] - diff(instance, near_hit)[f] / (m * k) + diff(instance, near_miss)[f] / (m * k)
   
Onde:
   - f: índice da feature
   - m: número de iterações (amostras)
   - k: número de vizinhos considerados
   - diff: diferença normalizada entre duas instâncias

INTERPRETAÇÃO:
- Peso positivo alto: feature separa bem instâncias de classes diferentes
- Peso negativo: feature não é discriminante
- Features com pesos maiores são mais relevantes

REFERÊNCIA:
Kononenko (1994) "Estimating Attributes: Analysis and Extensions of RELIEF"
Robnik-Šikonja & Kononenko (2003) "Theoretical and Empirical Analysis of ReliefF and RReliefF"

NOTA: Usaremos implementação do sklearn (ReliefF) através do pacote skrebate
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def calculate_relieff_score(X, y, n_neighbors=10, n_samples=100, verbose=True):
    """
    Calcula ReliefF score para cada feature usando implementação manual simplificada.
    
    NOTA: Esta é uma implementação simplificada do ReliefF.
    Para produção, recomenda-se usar a biblioteca skrebate.
    
    Parâmetros:
    -----------
    X : np.ndarray
        Matriz de features normalizadas [n_samples, n_features]
    y : np.ndarray
        Array de labels [n_samples]
    n_neighbors : int
        Número de vizinhos (k) a considerar
    n_samples : int
        Número de amostras aleatórias a processar
    verbose : bool
        Se True, imprime informações durante o cálculo
        
    Retorna:
    --------
    scores : np.ndarray
        Array com ReliefF score de cada feature [n_features]
    """
    n_instances, n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)
    
    if verbose:
        print(f"Calculando ReliefF Score (amostra de {n_samples} instâncias)...")
    
    # Inicializa pesos
    weights = np.zeros(n_features)
    
    # Limita número de amostras
    n_samples = min(n_samples, n_instances)
    
    # Seleciona índices aleatórios
    np.random.seed(42)  # Para reprodutibilidade
    random_indices = np.random.choice(n_instances, size=n_samples, replace=False)
    
    # Para cada amostra aleatória
    for i, idx in enumerate(random_indices):
        instance = X[idx]
        instance_class = y[idx]
        
        # Encontra near-hits (mesma classe)
        same_class_mask = (y == instance_class)
        same_class_indices = np.where(same_class_mask)[0]
        same_class_indices = same_class_indices[same_class_indices != idx]  # Remove própria instância
        
        if len(same_class_indices) == 0:
            continue
        
        # Calcula distâncias para instâncias da mesma classe
        same_class_data = X[same_class_indices]
        distances_hit = np.linalg.norm(same_class_data - instance, axis=1)
        
        # Seleciona k nearest hits
        k_hit = min(n_neighbors, len(same_class_indices))
        nearest_hit_indices = same_class_indices[np.argsort(distances_hit)[:k_hit]]
        
        # Atualiza pesos baseado em hits (penaliza diferenças dentro da mesma classe)
        for f in range(n_features):
            diff_hit = np.mean(np.abs(X[nearest_hit_indices, f] - instance[f]))
            weights[f] -= diff_hit / n_samples
        
        # Para cada classe diferente, encontra nearest misses
        for other_class in classes:
            if other_class == instance_class:
                continue
            
            # Encontra near-misses (classe diferente)
            other_class_mask = (y == other_class)
            other_class_indices = np.where(other_class_mask)[0]
            
            if len(other_class_indices) == 0:
                continue
            
            # Calcula distâncias para instâncias de outra classe
            other_class_data = X[other_class_indices]
            distances_miss = np.linalg.norm(other_class_data - instance, axis=1)
            
            # Seleciona k nearest misses
            k_miss = min(n_neighbors, len(other_class_indices))
            nearest_miss_indices = other_class_indices[np.argsort(distances_miss)[:k_miss]]
            
            # Calcula probabilidade da classe (peso para ReliefF multi-classe)
            prior_class = np.sum(y == other_class) / len(y)
            
            # Atualiza pesos baseado em misses (recompensa diferenças entre classes)
            for f in range(n_features):
                diff_miss = np.mean(np.abs(X[nearest_miss_indices, f] - instance[f]))
                weights[f] += prior_class * diff_miss / n_samples
    
    if verbose:
        print(f"ReliefF Scores calculados (média: {np.mean(weights):.4f})")
    
    return weights


def rank_features_relieff(relieff_scores, feature_names, top_k=10):
    """
    Rankeia features por ReliefF Score (ordem decrescente).
    
    Parâmetros:
    -----------
    relieff_scores : np.ndarray
        Array com ReliefF Score de cada feature [n_features]
    feature_names : list
        Lista com nomes das features
    top_k : int
        Número de melhores features a retornar
        
    Retorna:
    --------
    ranking : list of tuples
        Lista de tuplas (feature_name, score, rank) ordenada por score
    """
    # Ordenar índices por score (decrescente)
    sorted_indices = np.argsort(relieff_scores)[::-1]
    
    # Criar ranking
    ranking = []
    for rank, idx in enumerate(sorted_indices[:top_k], start=1):
        feature_name = feature_names[idx]
        score = relieff_scores[idx]
        ranking.append((feature_name, score, rank))
    
    return ranking


def print_relieff_ranking(ranking, title="ReliefF - Top Features"):
    """
    Imprime ranking de features formatado.
    
    Parâmetros:
    -----------
    ranking : list of tuples
        Lista de tuplas (feature_name, score, rank)
    title : str
        Título do ranking
    """
    print(f"\n{title}")
    print("-" * 60)
    for feature_name, score, rank in ranking:
        print(f"  {rank}. {feature_name:<35} {score:>8.4f}")
    print("-" * 60)
