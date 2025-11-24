"""
EXERCÍCIO 4.5: Implementação do Fisher Score
Fisher Score é uma técnica de seleção de features supervisionada que mede a 
capacidade discriminante de cada feature em relação às classes.

FÓRMULA:
Para cada feature j:
    Fisher_Score(j) = Σ_c n_c * (μ_cj - μ_j)² / Σ_c n_c * σ²_cj
    
Onde:
    - c: índice da classe
    - n_c: número de amostras da classe c
    - μ_cj: média da feature j na classe c
    - μ_j: média global da feature j
    - σ²_cj: variância da feature j na classe c

INTERPRETAÇÃO:
- Score alto: feature discrimina bem entre as classes (grande diferença entre médias, pequena variância intra-classe)
- Score baixo: feature não discrimina bem (pequena diferença entre médias, grande variância intra-classe)

REFERÊNCIA:
Gu et al. (2012) "Generalized Fisher Score for Feature Selection"
"""

import numpy as np


def calculate_fisher_score(X, y, verbose=True):
    """
    Calcula Fisher Score para cada feature.
    
    Parâmetros:
    -----------
    X : np.ndarray
        Matriz de features [n_samples, n_features]
    y : np.ndarray
        Array de labels [n_samples]
    verbose : bool
        Se True, imprime informações durante o cálculo
        
    Retorna:
    --------
    scores : np.ndarray
        Array com Fisher Score de cada feature [n_features]
    """
    n_samples, n_features = X.shape
    classes = np.unique(y)
    n_classes = len(classes)
    
    if verbose:
        print(f"Calculando Fisher Score para {n_features} features...")
    
    # Inicializa array de scores
    fisher_scores = np.zeros(n_features)
    
    # Média global de cada feature
    global_mean = np.mean(X, axis=0)  # [n_features]
    
    # Para cada feature
    for j in range(n_features):
        feature_col = X[:, j]
        
        numerator = 0.0  # Σ_c n_c * (μ_cj - μ_j)²
        denominator = 0.0  # Σ_c n_c * σ²_cj
        
        # Para cada classe
        for c in classes:
            # Amostras da classe c
            class_mask = (y == c)
            class_samples = feature_col[class_mask]
            n_c = len(class_samples)
            
            if n_c == 0:
                continue
            
            # Média da feature j na classe c
            class_mean = np.mean(class_samples)
            
            # Variância da feature j na classe c
            class_variance = np.var(class_samples)
            
            # Acumula numerador e denominador
            numerator += n_c * (class_mean - global_mean[j]) ** 2
            denominator += n_c * class_variance
        
        # Evita divisão por zero
        if denominator < 1e-10:
            fisher_scores[j] = 0.0
        else:
            fisher_scores[j] = numerator / denominator
    
    if verbose:
        print(f"Fisher Scores calculados (média: {np.mean(fisher_scores):.2f})")
    
    return fisher_scores


def rank_features_fisher(fisher_scores, feature_names, top_k=10):
    """
    Rankeia features por Fisher Score (ordem decrescente).
    
    Parâmetros:
    -----------
    fisher_scores : np.ndarray
        Array com Fisher Score de cada feature [n_features]
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
    sorted_indices = np.argsort(fisher_scores)[::-1]
    
    # Criar ranking
    ranking = []
    for rank, idx in enumerate(sorted_indices[:top_k], start=1):
        feature_name = feature_names[idx]
        score = fisher_scores[idx]
        ranking.append((feature_name, score, rank))
    
    return ranking


def print_fisher_ranking(ranking, title="Fisher Score - Top Features"):
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
        print(f"  {rank}. {feature_name:<35} {score:>8.2f}")
    print("-" * 60)
