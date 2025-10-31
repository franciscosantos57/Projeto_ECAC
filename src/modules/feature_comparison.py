"""
EXERCÍCIO 4.6: Comparação entre Fisher Score e ReliefF
Identifica as 10 melhores features de acordo com cada método e compara os resultados.

ANÁLISE COMPARATIVA:
- Fisher Score: Baseado em diferenças de médias e variâncias entre classes
- ReliefF: Baseado em diferenças entre instâncias vizinhas

Ambos são métodos supervisionados de seleção de features, mas com abordagens diferentes:
- Fisher Score: Estatístico, global, assume distribuições
- ReliefF: Baseado em instâncias, local, não assume distribuições
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from src.modules.fisher_score import calculate_fisher_score, rank_features_fisher, print_fisher_ranking
from src.modules.relieff_selection import calculate_relieff_score, rank_features_relieff, print_relieff_ranking


def compare_feature_selection_methods(X, y, feature_names, top_k=10, 
                                     relieff_neighbors=10, relieff_samples=100,
                                     verbose=True):
    """
    Compara Fisher Score e ReliefF na seleção das top-k features.
    
    Parâmetros:
    -----------
    X : np.ndarray
        Matriz de features [n_samples, n_features]
    y : np.ndarray
        Array de labels [n_samples]
    feature_names : list
        Lista com nomes das features
    top_k : int
        Número de melhores features a identificar
    relieff_neighbors : int
        Número de vizinhos para ReliefF
    relieff_samples : int
        Número de amostras para ReliefF
    verbose : bool
        Se True, imprime informações
        
    Retorna:
    --------
    results : dict
        Dicionário com resultados de ambos os métodos
    """
    if verbose:
        print(f"Comparando Fisher Score vs ReliefF (top-{top_k} features)")
    
    # Fisher Score (funciona com features não normalizadas)
    fisher_scores = calculate_fisher_score(X, y, verbose=verbose)
    fisher_ranking = rank_features_fisher(fisher_scores, feature_names, top_k=top_k)
    
    # ReliefF (requer features normalizadas para cálculo de distâncias)
    # Normalizar features com Z-Score
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    
    relieff_scores = calculate_relieff_score(X_normalized, y, 
                                            n_neighbors=relieff_neighbors,
                                            n_samples=relieff_samples,
                                            verbose=verbose)
    relieff_ranking = rank_features_relieff(relieff_scores, feature_names, top_k=top_k)
    
    # Comparação lado a lado
    if verbose:
        compare_rankings(fisher_ranking, relieff_ranking, top_k=top_k)
    
    # Retorna resultados
    results = {
        'fisher_scores': fisher_scores,
        'fisher_ranking': fisher_ranking,
        'relieff_scores': relieff_scores,
        'relieff_ranking': relieff_ranking,
        'comparison': analyze_agreement(fisher_ranking, relieff_ranking)
    }
    
    return results


def compare_rankings(fisher_ranking, relieff_ranking, top_k=10):
    """
    Compara os rankings de Fisher Score e ReliefF lado a lado.
    
    Parâmetros:
    -----------
    fisher_ranking : list of tuples
        Ranking do Fisher Score
    relieff_ranking : list of tuples
        Ranking do ReliefF
    top_k : int
        Número de features a comparar
    """
    print(f"\nComparação Fisher vs ReliefF:")
    print("-" * 80)
    print(f"{'#':<3} {'Fisher Feature':<30} {'Score':>8}  |  {'ReliefF Feature':<30} {'Score':>8}")
    print("-" * 80)
    
    for i in range(top_k):
        fisher_feat, fisher_score, _ = fisher_ranking[i]
        relieff_feat, relieff_score, _ = relieff_ranking[i]
        
        # Marca se a feature aparece em ambos os rankings
        marker = " ★" if fisher_feat in [f[0] for f in relieff_ranking] else ""
        
        print(f"{i+1:<3} {fisher_feat:<30} {fisher_score:>8.2f}  |  {relieff_feat:<30} {relieff_score:>8.4f}{marker}")
    
    print("-" * 80)
    print("★ = Feature também aparece no top-10 do outro método")


def analyze_agreement(fisher_ranking, relieff_ranking):
    """
    Analisa concordância entre os dois métodos.
    
    Parâmetros:
    -----------
    fisher_ranking : list of tuples
        Ranking do Fisher Score
    relieff_ranking : list of tuples
        Ranking do ReliefF
        
    Retorna:
    --------
    agreement : dict
        Dicionário com métricas de concordância
    """
    # Extrai nomes das features
    fisher_features = set([f[0] for f in fisher_ranking])
    relieff_features = set([f[0] for f in relieff_ranking])
    
    # Calcula intersecção
    common_features = fisher_features.intersection(relieff_features)
    
    # Calcula métricas
    n_common = len(common_features)
    agreement_rate = n_common / len(fisher_features)
    
    # Analisa posições relativas das features comuns
    position_differences = []
    for feature in common_features:
        fisher_pos = next(i for i, (f, _, _) in enumerate(fisher_ranking) if f == feature)
        relieff_pos = next(i for i, (f, _, _) in enumerate(relieff_ranking) if f == feature)
        position_differences.append(abs(fisher_pos - relieff_pos))
    
    avg_position_diff = np.mean(position_differences) if position_differences else 0
    
    # Imprime análise
    print(f"\nConcordância: {n_common}/{len(fisher_features)} features comuns ({agreement_rate * 100:.0f}%)")
    
    if common_features:
        print(f"Features consensuais:")
        for feature in sorted(common_features):
            fisher_pos = next(j + 1 for j, (f, _, _) in enumerate(fisher_ranking) if f == feature)
            relieff_pos = next(j + 1 for j, (f, _, _) in enumerate(relieff_ranking) if f == feature)
            print(f"  • {feature} (Fisher: #{fisher_pos}, ReliefF: #{relieff_pos})")
    
    return {
        'n_common': n_common,
        'common_features': list(common_features),
        'agreement_rate': agreement_rate,
        'avg_position_diff': avg_position_diff
    }


def extract_selected_features(X, feature_names, selected_feature_names):
    """
    EXERCÍCIO 4.6.1: Extrai apenas as features selecionadas da matriz completa.
    
    Parâmetros:
    -----------
    X : np.ndarray
        Matriz de features completa [n_samples, n_features]
    feature_names : list
        Lista com todos os nomes das features
    selected_feature_names : list
        Lista com nomes das features selecionadas
        
    Retorna:
    --------
    X_selected : np.ndarray
        Matriz com apenas as features selecionadas [n_samples, n_selected]
    selected_indices : list
        Índices das features selecionadas
    """
    # Encontra índices das features selecionadas
    selected_indices = [i for i, name in enumerate(feature_names) 
                       if name in selected_feature_names]
    
    # Extrai colunas correspondentes
    X_selected = X[:, selected_indices]
    
    return X_selected, selected_indices


def demonstrate_feature_extraction(X, feature_names, fisher_ranking, relieff_ranking, 
                                   sample_idx=0, verbose=True):
    """
    EXERCÍCIO 4.6.1: Demonstra extração de features para um instante específico.
    
    Parâmetros:
    -----------
    X : np.ndarray
        Matriz de features completa
    feature_names : list
        Lista com nomes das features
    fisher_ranking : list
        Ranking do Fisher Score
    relieff_ranking : list
        Ranking do ReliefF
    sample_idx : int
        Índice do instante a exemplificar
    verbose : bool
        Se True, imprime exemplo
        
    Retorna:
    --------
    example : dict
        Dicionário com exemplo de extração
    """
    # Extrai top-10 features de cada método
    fisher_features = [f[0] for f in fisher_ranking[:10]]
    relieff_features = [f[0] for f in relieff_ranking[:10]]
    
    # Features da união de ambos os métodos
    union_features = list(set(fisher_features + relieff_features))
    
    # Extrai features selecionadas
    X_fisher, fisher_indices = extract_selected_features(X, feature_names, fisher_features)
    X_relieff, relieff_indices = extract_selected_features(X, feature_names, relieff_features)
    X_union, union_indices = extract_selected_features(X, feature_names, union_features)
    
    # Exemplo para o instante especificado
    sample_original = X[sample_idx]
    sample_fisher = X_fisher[sample_idx]
    sample_relieff = X_relieff[sample_idx]
    sample_union = X_union[sample_idx]
    
    if verbose:
        print(f"\nExemplo de extração de features (instante {sample_idx}):")
        print("-" * 80)
        print(f"Dimensionalidade original: {X.shape[1]} features")
        print(f"Após seleção Fisher:      {X_fisher.shape[1]} features")
        print(f"Após seleção ReliefF:     {X_relieff.shape[1]} features")
        print(f"União de ambos:            {X_union.shape[1]} features")
        print(f"\nRedução: {X.shape[1]} → {X_union.shape[1]} features "
              f"({100 * (1 - X_union.shape[1] / X.shape[1]):.1f}% compressão)")
    
    return {
        'sample_idx': sample_idx,
        'original_features': sample_original,
        'fisher_features': sample_fisher,
        'relieff_features': sample_relieff,
        'union_features': sample_union,
        'fisher_names': fisher_features,
        'relieff_names': relieff_features,
        'union_names': union_features,
        'fisher_indices': fisher_indices,
        'relieff_indices': relieff_indices,
        'union_indices': union_indices
    }


def analyze_selection_approach(fisher_ranking, relieff_ranking, X_shape, verbose=True):
    """
    EXERCÍCIO 4.6.2: Analisa vantagens e limitações da seleção de features.
    
    Parâmetros:
    -----------
    fisher_ranking : list
        Ranking do Fisher Score
    relieff_ranking : list
        Ranking do ReliefF
    X_shape : tuple
        Shape da matriz original (n_samples, n_features)
    verbose : bool
        Se True, imprime análise resumida
        
    Retorna:
    --------
    analysis : dict
        Dicionário com análise da abordagem
    """
    n_samples, n_features = X_shape
    
    # Métricas de concordância
    fisher_features = set([f[0] for f in fisher_ranking[:10]])
    relieff_features = set([f[0] for f in relieff_ranking[:10]])
    common = fisher_features.intersection(relieff_features)
    union = fisher_features.union(relieff_features)
    
    agreement_rate = len(common) / 10
    union_size = len(union)
    
    if verbose:
        print(f"\nAnálise da abordagem de seleção:")
        print("-" * 80)
        print(f"Features comuns: {len(common)}/10 ({agreement_rate * 100:.0f}% concordância)")
        print(f"União: {union_size} features únicas entre ambos os métodos")
        print(f"Redução: {n_features} → {union_size} features "
              f"({100 * (1 - union_size / n_features):.1f}% compressão)")
    
    return {
        'n_original': n_features,
        'n_fisher': 10,
        'n_relieff': 10,
        'n_common': len(common),
        'n_union': union_size,
        'agreement_rate': agreement_rate,
        'compression_rate': 1 - union_size / n_features,
        'common_features': list(common),
        'fisher_only': list(fisher_features - relieff_features),
        'relieff_only': list(relieff_features - fisher_features)
    }
