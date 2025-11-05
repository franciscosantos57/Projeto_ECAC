"""
EXERCÍCIO 3.7.1: DBSCAN para deteção de outliers.
Identifica clusters densos e outliers usando sklearn.cluster.DBSCAN.
"""

import os
import numpy as np
from sklearn.cluster import DBSCAN

from src.utils.sensor_calculations import calculate_sensor_modules, zscore_normalization
from src.utils.plot_3d import create_3d_visualization_dbscan


def sample_data(data, sample_fraction=50, random_state=42):
    """
    Amostra aleatória dos dados para processamento mais rápido.
    
    Args:
        data: Array numpy com os dados completos
        sample_fraction: Fração dos dados a usar (1/sample_fraction)
                        Default: 50 (usa 1/50 dos dados)
        random_state: Seed para reprodutibilidade
        
    Returns:
        Array numpy com amostra dos dados
    """
    np.random.seed(random_state)
    sample_size = len(data) // sample_fraction
    indices = np.random.choice(len(data), size=sample_size, replace=False)
    return data[indices]


def detect_outliers_dbscan(data, eps=0.5, min_samples=5, use_modules=True, normalize=True):
    """
    EXERCÍCIO 3.7.1: Deteta outliers usando algoritmo DBSCAN.
    
    O DBSCAN agrupa pontos densos em clusters e marca pontos isolados como outliers.
    
    Parâmetros:
        eps: Raio máximo da vizinhança (epsilon)
        min_samples: Número mínimo de pontos para formar um cluster denso
        use_modules: Se True, usa módulos dos sensores; se False, usa dados brutos
        normalize: Se True, normaliza os dados antes do clustering
        
    Args:
        data: Array numpy com dados de sensores
        eps: float, raio epsilon para DBSCAN
        min_samples: int, mínimo de amostras por cluster
        use_modules: bool, usar módulos dos sensores (3D)
        normalize: bool, normalizar dados
        
    Returns:
        dict com:
            - 'labels': Array com labels dos clusters (-1 = outlier)
            - 'outliers': Array booleano (True = outlier)
            - 'n_clusters': Número de clusters encontrados
            - 'n_outliers': Número de outliers
            - 'outlier_percentage': Percentagem de outliers
            - 'features': Features usadas (normalizadas se normalize=True)
    """
    # Preparar features (mesmo método do K-Means para consistência)
    if use_modules:
        # Usar módulos dos sensores (3D) - exatamente como K-Means
        modules_dict = calculate_sensor_modules(data)
        features = np.column_stack([
            modules_dict['acc_module'],
            modules_dict['gyro_module'],
            modules_dict['mag_module']
        ])
    else:
        # Usar dados brutos (9D) - colunas 2-10
        features = data[:, 1:10]
    
    # Normalizar usando Z-Score customizado (mesmo que K-Means)
    if normalize:
        features_scaled = zscore_normalization(features)
    else:
        features_scaled = features
    
    # Aplicar DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = dbscan.fit_predict(features_scaled)
    
    # Identificar outliers (label = -1)
    outliers = labels == -1
    n_outliers = np.sum(outliers)
    
    # Número de clusters (excluindo outliers)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Percentagem de outliers
    outlier_percentage = (n_outliers / len(data)) * 100
    
    return {
        'labels': labels,
        'outliers': outliers,
        'n_clusters': n_clusters,
        'n_outliers': n_outliers,
        'outlier_percentage': outlier_percentage,
        'features': features_scaled if normalize else features,
        'normalized_features': features_scaled,  # Sempre retorna features normalizadas
        'eps': eps,
        'min_samples': min_samples
    }


def analyze_dbscan_outliers(data, eps_values, min_samples_values, create_plots=True):
    """
    Analisa outliers usando DBSCAN com diferentes combinações de parâmetros.
    Análogo ao analyze_kmeans_outliers do exercício 3.6/3.7.
    
    Args:
        data: Array numpy com dados completos
        eps_values: Lista de valores epsilon para testar
        min_samples_values: Lista de valores min_samples para testar
        create_plots: Se True, cria visualizações 3D
        
    Returns:
        Dict com resultados para cada combinação de parâmetros
    """
    print(f"\n{'=' * 60}")
    print("DBSCAN clustering:")
    print(f"{'=' * 60}")
    
    # Usar 1/50 dos dados para evitar problemas de memória
    # DBSCAN é muito pesado em memória (matriz de distâncias N×N)
    data_sample = sample_data(data, sample_fraction=50, random_state=42)
    print(f"\nAmostra: {len(data_sample):,} pontos (1/50 de {len(data):,} totais - DBSCAN requer menos que K-Means)")
    
    results = {}
    
    print(f"Testando {len(eps_values) * len(min_samples_values)} combinações de parâmetros\n")
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            key = f"eps{eps}_ms{min_samples}"
            
            # Detectar outliers
            result = detect_outliers_dbscan(
                data_sample,
                eps=eps,
                min_samples=min_samples,
                use_modules=True,
                normalize=True
            )
            
            results[key] = result
            
            print(f"eps={eps}, ms={min_samples}: {result['n_clusters']} clusters, {result['n_outliers']:,} outliers ({result['outlier_percentage']:.1f}%)")
            
            # Criar visualização se solicitado
            if create_plots:
                fig = create_3d_visualization_dbscan(
                    data_sample,
                    result,
                    title_suffix=f"Amostra: {len(data_sample):,} pontos"
                )
                
                filename = f"plots/exercicio_3.7.1_dbscan/dbscan_3d_eps{eps}_ms{min_samples}.png"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                fig.savefig(filename, dpi=150, bbox_inches='tight')
                del fig  # Liberta memória
                print(f"  Gráfico guardado: {filename}\n")
    
    return results


def summarize_dbscan_analysis(results):
    """
    Resumo da análise DBSCAN com diferentes parâmetros.
    
    Args:
        results: Dict com resultados de analyze_dbscan_outliers()
    """
    print("\nResumo DBSCAN:")
    for key, result in results.items():
        print(f"  eps={result['eps']}, ms={result['min_samples']}: {result['n_clusters']} clusters, {result['outlier_percentage']:.1f}% outliers")
