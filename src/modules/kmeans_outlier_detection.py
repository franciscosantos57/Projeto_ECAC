"""
EXERCÍCIO 3.6 e 3.7: K-Means para deteção de outliers.
Agrupa dados em clusters e identifica outliers baseado em distância aos centroides.
"""

import numpy as np
import os

from src.utils.sensor_calculations import calculate_sensor_modules, zscore_normalization
from src.utils.plot_3d import create_3d_visualization_kmeans, create_3d_visualization_kmeans_zoom


def kmeans_clustering(data, n_clusters, max_iter=100, random_state=42):
    """
    EXERCÍCIO 3.6: Implementação do algoritmo K-Means.
    
    Algoritmo iterativo que agrupa dados em k clusters:
    1. Inicializa k centroides aleatórios
    2. Atribui cada ponto ao centroide mais próximo
    3. Recalcula centroides como média dos pontos
    4. Repete até convergência
    """
    np.random.seed(random_state)
    n_samples, n_features = data.shape
    
    # 1. Inicialização: seleciona n_clusters pontos aleatórios como centroides iniciais
    indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = data[indices].copy()
    
    labels = np.zeros(n_samples, dtype=int)
    
    # Iteração do algoritmo
    for iteration in range(max_iter):
        old_centroids = centroids.copy()
        
        # 2. Atribuição: cada ponto ao cluster do centroide mais próximo
        # Calcula distância euclidiana de cada ponto a cada centroide
        distances_to_centroids = np.zeros((n_samples, n_clusters))
        for k in range(n_clusters):
            # Distância euclidiana: ||x - c_k||
            distances_to_centroids[:, k] = np.sqrt(np.sum((data - centroids[k])**2, axis=1))
        
        # Atribui cada ponto ao cluster mais próximo
        labels = np.argmin(distances_to_centroids, axis=1)
        
        # 3. Atualização: recalcula centroides como média dos pontos de cada cluster
        for k in range(n_clusters):
            cluster_points = data[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
        
        # Verifica convergência: centroides não mudaram
        if np.allclose(centroids, old_centroids):
            break
    
    # Calcula distâncias finais de cada ponto ao seu centroide
    distances = np.zeros(n_samples)
    for i in range(n_samples):
        distances[i] = np.sqrt(np.sum((data[i] - centroids[labels[i]])**2))
    
    # Calcula inércia: soma das distâncias quadráticas
    inertia = np.sum(distances**2)
    
    return {
        'labels': labels,
        'centroids': centroids,
        'distances': distances,
        'inertia': inertia,
        'n_iter': iteration + 1
    }

def detect_outliers_kmeans(data, n_clusters_list=[3, 5, 7, 10], use_modules=True):
    """
    EXERCÍCIO 3.7: Detecta outliers usando K-Means clustering.
    
    A deteção de outliers por K-Means baseia-se na distância de cada ponto
    ao centroide do seu cluster. Pontos muito distantes do centroide são
    considerados outliers.
    
    Utiliza o método IQR (Interquartile Range - Tukey) para identificar outliers:
    - Calcula Q1 (percentil 25) e Q3 (percentil 75) das distâncias
    - IQR = Q3 - Q1
    - Threshold = Q3 + 1.5 × IQR
    - Outliers são pontos com distância > threshold
    
    Pode usar:
    - Espaço dos módulos: [acc_module, gyro_module, mag_module]
    - Espaço original: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z]
    
    NOTA: Os dados são normalizados usando Z-Score antes do clustering.
    
    Args:
        data (numpy.ndarray): Dados combinados de todos os participantes
        n_clusters_list (list): Lista de números de clusters a testar
        use_modules (bool): True para usar módulos, False para valores originais
    
    Returns:
        dict: Resultados para cada valor de n_clusters
    """
    # Prepara os dados para clustering
    if use_modules:
        modules = calculate_sensor_modules(data)
        clustering_data = np.column_stack([
            modules['acc_module'],
            modules['gyro_module'],
            modules['mag_module']
        ])
        space_name = "módulos"
    else:
        clustering_data = data[:, 1:10]
        space_name = "valores originais"
    
    print(f"Dimensão: {clustering_data.shape[0]} amostras × {clustering_data.shape[1]} features")
    
    # NORMALIZAÇÃO Z-SCORE
    clustering_data_normalized = zscore_normalization(clustering_data)
    
    results = {}
    
    for n_clusters in n_clusters_list:        
        # Aplica K-Means nos dados normalizados
        kmeans_result = kmeans_clustering(clustering_data_normalized, n_clusters)
        
        # Deteta outliers baseado nas distâncias usando método IQR (Tukey)
        distances = kmeans_result['distances']
        
        # Calcula quartis e IQR
        Q1 = np.percentile(distances, 25)
        Q3 = np.percentile(distances, 75)
        IQR = Q3 - Q1
        
        # Threshold: Q3 + 1.5 * IQR (método de Tukey para outliers)
        threshold = Q3 + 1.5 * IQR
        outliers_mask = distances > threshold
        
        # Calcula também estatísticas descritivas para referência
        mean_distance = np.mean(distances)
        median_distance = np.median(distances)
        
        n_outliers = np.sum(outliers_mask)
        outlier_percentage = (n_outliers / len(data)) * 100
        
        print(f"k={n_clusters}: {n_outliers} outliers ({outlier_percentage:.1f}%), inércia={kmeans_result['inertia']:.0f}")
        
        results[n_clusters] = {
            'kmeans_result': kmeans_result,
            'outliers_mask': outliers_mask,
            'n_outliers': n_outliers,
            'outlier_percentage': outlier_percentage,
            'threshold': threshold,
            'mean_distance': mean_distance,
            'median_distance': median_distance,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'normalized_data': clustering_data_normalized
        }
    
    return results


def analyze_kmeans_outliers(data, cluster_range=[3, 5, 7, 10], use_modules=True, 
                            create_plots=True, output_dir_normal="plots", output_dir_zoom="plots"):
    """
    Análise completa de outliers usando K-Means com diferentes números de clusters.
    
    Args:
        data (numpy.ndarray): Dados combinados de todos os participantes
        cluster_range (list): Lista de números de clusters a testar
        use_modules (bool): True para usar módulos, False para valores originais
        create_plots (bool): True para criar visualizações 3D
        output_dir_normal (str): Diretório para gráficos normais
        output_dir_zoom (str): Diretório para gráficos zoom
    
    Returns:
        dict: Resultados completos da análise
    """
    print("\n" + "="*60)
    print("K-Means clustering:")
    print("="*60)
    
    # Usar apenas 1/10 dos dados (10% do dataset)
    sample_size = len(data) // 10
    np.random.seed(42)
    indices = np.random.choice(len(data), sample_size, replace=False)
    data_sample = data[indices]
    print(f"\nUsando amostra de {sample_size:,} pontos (1/10 de {len(data):,} totais)")
    
    # Deteta outliers com K-Means
    results = detect_outliers_kmeans(data_sample, n_clusters_list=cluster_range, 
                                    use_modules=use_modules)
    
    # Cria visualizações 3D
    if create_plots:
        print("\nCriando visualizações 3D...")
        for n_clusters in cluster_range:
            # Gráfico normal (4 ângulos com outliers)
            create_3d_visualization_kmeans(
                data_sample, 
                results[n_clusters]['kmeans_result'],
                results[n_clusters]['outliers_mask'],
                n_clusters,
                use_modules=use_modules,
                normalized_data=results[n_clusters]['normalized_data'],
                output_dir=output_dir_normal
            )
            
            # Gráfico ZOOM (4 ângulos SEM outliers) - arquivo separado
            create_3d_visualization_kmeans_zoom(
                data_sample, 
                results[n_clusters]['kmeans_result'],
                results[n_clusters]['outliers_mask'],
                n_clusters,
                use_modules=use_modules,
                normalized_data=results[n_clusters]['normalized_data'],
                output_dir=output_dir_zoom
            )
    
    # Resumo minimalista
    print("\nResumo K-Means:")
    for n_clusters in sorted(results.keys()):
        r = results[n_clusters]
        print(f"  k={n_clusters}: {r['n_outliers']} outliers ({r['outlier_percentage']:.1f}%)")
    
    # Retorna os resultados junto com os dados amostrados
    return {'results': results, 'data_sample': data_sample}

def compare_with_zscore(data, kmeans_results, k_zscore=3):
    """
    Compara os resultados do K-Means com o método Z-Score.
    
    Args:
        data (numpy.ndarray): Dados originais (pode ser amostra ou completo)
        kmeans_results (dict): Resultados do K-Means (pode ter 'results' e 'data_sample')
        k_zscore (float): Valor de k para Z-Score
    """
    from .zscore_outlier_detection import detect_outliers_zscore
    
    # Se kmeans_results tem estrutura nova, extrair dados
    if 'results' in kmeans_results and 'data_sample' in kmeans_results:
        actual_results = kmeans_results['results']
        data_to_use = kmeans_results['data_sample']
        print("\n(Usando amostra de dados para comparação consistente)")
    else:
        actual_results = kmeans_results
        data_to_use = data
    
    # Calcula módulos
    modules = calculate_sensor_modules(data_to_use)
    
    # Deteta outliers com Z-Score para cada módulo
    acc_outliers = detect_outliers_zscore(modules['acc_module'], k_zscore)
    gyro_outliers = detect_outliers_zscore(modules['gyro_module'], k_zscore)
    mag_outliers = detect_outliers_zscore(modules['mag_module'], k_zscore)
    
    # Combina outliers (se é outlier em qualquer sensor)
    zscore_outliers = acc_outliers | gyro_outliers | mag_outliers
    n_zscore = np.sum(zscore_outliers)
    
    print(f"\nComparação K-Means vs Z-Score (k={k_zscore}):")
    print(f"  Z-Score: {n_zscore} outliers ({n_zscore/len(data_to_use)*100:.1f}%)")
    
    for n_clusters in sorted(actual_results.keys()):
        kmeans_outliers = actual_results[n_clusters]['outliers_mask']
        n_kmeans = np.sum(kmeans_outliers)
        overlap = np.sum(kmeans_outliers & zscore_outliers)
        print(f"  K-Means (k={n_clusters}): {n_kmeans} outliers, {overlap} em comum")
