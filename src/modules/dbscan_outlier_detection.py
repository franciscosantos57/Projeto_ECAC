"""
EXERCÍCIO 3.7.1: DBSCAN para Deteção de Outliers
Implementação usando sklearn.cluster.DBSCAN
Análogo ao exercício 3.6/3.7 mas com DBSCAN

O DBSCAN identifica:
- Clusters densos (core points e border points)
- Outliers/Noise (pontos não pertencentes a nenhum cluster)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend não-GUI
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN

from src.utils.sensor_calculations import calculate_sensor_modules, zscore_normalization


def sample_data(data, sample_size=50000, random_state=42):
    """
    Amostra aleatória dos dados para processamento mais rápido.
    
    Args:
        data: Array numpy com os dados completos
        sample_size: Número de amostras a extrair
        random_state: Seed para reprodutibilidade
        
    Returns:
        Array numpy com amostra dos dados
    """
    np.random.seed(random_state)
    n_samples = min(sample_size, len(data))
    indices = np.random.choice(len(data), size=n_samples, replace=False)
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
        modules_dict = {
            'acc_module': np.sqrt(data[:, 1]**2 + data[:, 2]**2 + data[:, 3]**2),
            'gyro_module': np.sqrt(data[:, 4]**2 + data[:, 5]**2 + data[:, 6]**2),
            'mag_module': np.sqrt(data[:, 7]**2 + data[:, 8]**2 + data[:, 9]**2)
        }
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


def create_3d_visualization_dbscan(data, result, title_suffix=""):
    """
    Cria visualização 3D com 4 ângulos diferentes (VOLTOU AO NORMAL).
    
    Mostra:
    - Clusters em cores MUITO DISTINTAS (mesmo padrão do K-Means)
    - Outliers em VERMELHO
    - 4 ângulos NORMAIS (todos com outliers)
    
    IMPORTANTE: Usa dados normalizados (mesma escala usada pelo DBSCAN)
    
    Args:
        data: Array numpy com dados originais (não usado mais)
        result: Dicionário retornado por detect_outliers_dbscan()
        title_suffix: Sufixo para o título do gráfico
    """
    # Usar features normalizadas do resultado (mesma escala do DBSCAN)
    modules = result['normalized_features']
    labels = result['labels']
    outliers_mask = result['outliers']
    n_clusters = result['n_clusters']
    
    # Extrair coordenadas
    x = modules[:, 0]
    y = modules[:, 1]
    z = modules[:, 2]
    
    xlabel = 'Módulo Acelerómetro (Z-Score)'
    ylabel = 'Módulo Giroscópio (Z-Score)'
    zlabel = 'Módulo Magnetómetro (Z-Score)'
    
    # Criar figura com 4 subplots (2x2)
    fig = plt.figure(figsize=(20, 16))
    
    # Define cores DISTINTAS para clusters (MESMO ESQUEMA DO K-MEANS)
    if n_clusters <= 3:
        color_list = ['blue', 'green', 'orange']
    elif n_clusters <= 5:
        color_list = ['blue', 'green', 'orange', 'purple', 'brown']
    elif n_clusters <= 7:
        color_list = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
    else:
        # Para mais clusters, usa tab20 excluindo vermelhos
        cmap = plt.cm.tab20(np.linspace(0, 1, 20))
        color_indices = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19]
        color_list = [cmap[i] for i in color_indices[:n_clusters]]
    
    colors = color_list[:n_clusters]
    
    # Função auxiliar para plotar vista NORMAL (com outliers)
    def plot_3d_view(ax, elev, azim, title_suffix_view):
        # 1. Outliers PRIMEIRO (ficam atrás) - zorder=1
        if np.sum(outliers_mask) > 0:
            ax.scatter(x[outliers_mask], y[outliers_mask], z[outliers_mask],
                      c='red', label=f'Outliers ({np.sum(outliers_mask)})', 
                      alpha=0.4, s=20, marker='x', linewidths=1.5, 
                      zorder=1, depthshade=True)
        
        # 2. Pontos normais por cluster (ficam no meio) - zorder=2
        for k in range(n_clusters):
            cluster_mask = (labels == k) & (~outliers_mask)
            if np.sum(cluster_mask) > 0:
                ax.scatter(x[cluster_mask], y[cluster_mask], z[cluster_mask],
                          c=[colors[k]], label=f'Cluster {k+1}', 
                          alpha=0.5, s=8, edgecolors='none',
                          zorder=2, depthshade=True)
        
        ax.set_xlabel(xlabel, fontsize=10, labelpad=8, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, labelpad=8, fontweight='bold')
        ax.set_zlabel(zlabel, fontsize=10, labelpad=12, fontweight='bold')
        ax.set_title(f'{title_suffix_view}\nOutliers: {np.sum(outliers_mask)} '
                    f'({(np.sum(outliers_mask)/len(modules)*100):.2f}%)',
                    fontsize=11, fontweight='bold', pad=10)
        
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # 4 subplots NORMAIS (todos com outliers)
    ax1 = fig.add_subplot(221, projection='3d')
    plot_3d_view(ax1, elev=20, azim=45, title_suffix_view='Ângulo 1 (Frontal)')
    
    ax2 = fig.add_subplot(222, projection='3d')
    plot_3d_view(ax2, elev=20, azim=135, title_suffix_view='Ângulo 2 (Lateral Direita)')
    
    ax3 = fig.add_subplot(223, projection='3d')
    plot_3d_view(ax3, elev=20, azim=225, title_suffix_view='Ângulo 3 (Traseira)')
    
    ax4 = fig.add_subplot(224, projection='3d')
    plot_3d_view(ax4, elev=20, azim=315, title_suffix_view='Ângulo 4 (Lateral Esquerda)')
    
    # Legenda única e centralizada (MESMO ESTILO DO K-MEANS)
    from matplotlib.lines import Line2D
    
    legend_elements = []
    
    # Adiciona clusters
    for k in range(n_clusters):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=colors[k], markersize=8,
                                     label=f'Cluster {k+1}'))
    
    # Adiciona outliers
    if np.sum(outliers_mask) > 0:
        legend_elements.append(Line2D([0], [0], marker='x', color='w', 
                                     markerfacecolor='red', markeredgecolor='red',
                                     markersize=7, markeredgewidth=1.5,
                                     label=f'Outliers ({np.sum(outliers_mask)})'))
    
    # Título geral NO TOPO
    fig.suptitle(f'DBSCAN Clustering (eps={result["eps"]}, min_samples={result["min_samples"]})',
                fontsize=15, fontweight='bold', y=0.98)
    
    # Legenda ABAIXO dos gráficos
    fig.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, 0.02), ncol=min(n_clusters + 1, 7),
              fontsize=10, framealpha=0.95, edgecolor='black', 
              fancybox=True, shadow=True)
    
    # Ajusta layout
    plt.subplots_adjust(top=0.94, bottom=0.08, left=0.05, right=0.95, 
                       hspace=0.25, wspace=0.25)
    
    return fig


def analyze_dbscan_outliers(data, eps_values, min_samples_values, sample_size=None, create_plots=True):
    """
    Analisa outliers usando DBSCAN com diferentes combinações de parâmetros.
    Análogo ao analyze_kmeans_outliers do exercício 3.6/3.7.
    
    Args:
        data: Array numpy com dados completos
        eps_values: Lista de valores epsilon para testar
        min_samples_values: Lista de valores min_samples para testar
        sample_size: Tamanho da amostra (None = usar 1/25 dos dados, igual K-Means)
        create_plots: Se True, cria visualizações 3D
        
    Returns:
        Dict com resultados para cada combinação de parâmetros
    """
    print(f"\n{'=' * 60}")
    print("DBSCAN clustering:")
    print(f"{'=' * 60}")
    
    # Usar 1/50 dos dados para evitar problemas de memória
    # DBSCAN é muito pesado em memória (matriz de distâncias N×N)
    if sample_size is None:
        sample_size = len(data) // 50  # 2% do dataset
    
    print(f"Amostra: {sample_size:,} pontos (1/50 dos dados - DBSCAN requer menos que K-Means)")
    np.random.seed(42)
    indices = np.random.choice(len(data), sample_size, replace=False)
    data_sample = data[indices]
    
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
                import os
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                fig.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Gráfico salvo: {filename}\n")
    
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
