"""
Funções auxiliares para criar diversos tipos de visualizações 3D.
Inclui visualizações para K-Means e DBSCAN com diferentes perspetivas.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_3d_scatter(ax, features, colors, title, labels=None, alpha=0.5, s=10):
    """
    Cria um scatter plot 3D básico.
    
    Args:
        ax: Eixo matplotlib 3D
        features: Array [n_samples, 3] com coordenadas x, y, z
        colors: Cores dos pontos
        title: Título do gráfico
        labels: Labels dos pontos (opcional)
        alpha: Transparência dos pontos
        s: Tamanho dos pontos
    """
    scatter = ax.scatter(
        features[:, 0], features[:, 1], features[:, 2],
        c=colors, s=s, alpha=alpha, edgecolors='k', linewidth=0.1
    )
    
    ax.set_xlabel('Módulo Acelerómetro', fontsize=10)
    ax.set_ylabel('Módulo Giroscópio', fontsize=10)
    ax.set_zlabel('Módulo Magnetómetro', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    return scatter


def create_3d_visualization_kmeans(data, kmeans_result, outliers_mask, n_clusters, 
                                   zoom=False, title_suffix=""):
    """
    Cria visualização 3D para resultados do K-Means.
    
    Args:
        data: Dados originais
        kmeans_result: Resultado do K-Means
        outliers_mask: Array booleano de outliers
        n_clusters: Número de clusters
        zoom: Se True, faz zoom na região de outliers
        title_suffix: Texto adicional para o título
    
    Returns:
        matplotlib.figure.Figure: Figura criada
    """
    from src.utils.sensor_calculations import calculate_sensor_modules
    
    # Calcula módulos dos sensores
    modules = calculate_sensor_modules(data)
    features = np.column_stack([
        modules['acc_module'],
        modules['gyro_module'],
        modules['mag_module']
    ])
    
    # Cria figura
    fig = plt.figure(figsize=(16, 6))
    
    # Plot 1: Clusters coloridos
    ax1 = fig.add_subplot(131, projection='3d')
    colors_clusters = plt.cm.tab10(kmeans_result['labels'] % 10)
    create_3d_scatter(
        ax1, features, colors_clusters,
        f'Clusters K-Means (k={n_clusters})'
    )
    
    # Adiciona centroides
    centroids = kmeans_result['centroids']
    ax1.scatter(
        centroids[:, 0], centroids[:, 1], centroids[:, 2],
        c='red', s=200, marker='X', edgecolors='black', linewidth=2,
        label='Centroides'
    )
    ax1.legend()
    
    # Plot 2: Outliers destacados
    ax2 = fig.add_subplot(132, projection='3d')
    colors_outliers = np.where(outliers_mask, 'red', 'blue')
    create_3d_scatter(
        ax2, features, colors_outliers,
        'Outliers (vermelho) vs Normais (azul)'
    )
    
    # Plot 3: Distâncias aos centroides
    ax3 = fig.add_subplot(133, projection='3d')
    distances = kmeans_result['distances']
    scatter3 = ax3.scatter(
        features[:, 0], features[:, 1], features[:, 2],
        c=distances, s=10, alpha=0.5, cmap='YlOrRd', edgecolors='k', linewidth=0.1
    )
    ax3.set_xlabel('Módulo Acelerómetro', fontsize=10)
    ax3.set_ylabel('Módulo Giroscópio', fontsize=10)
    ax3.set_zlabel('Módulo Magnetómetro', fontsize=10)
    ax3.set_title('Distância ao Centroide (cores quentes = maior)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter3, ax=ax3, label='Distância', shrink=0.5)
    
    # Aplica zoom se necessário
    if zoom:
        # Define limites baseados em outliers
        outlier_features = features[outliers_mask]
        if len(outlier_features) > 0:
            margin = 5
            for ax in [ax1, ax2, ax3]:
                ax.set_xlim([outlier_features[:, 0].min() - margin, 
                           outlier_features[:, 0].max() + margin])
                ax.set_ylim([outlier_features[:, 1].min() - margin, 
                           outlier_features[:, 1].max() + margin])
                ax.set_zlim([outlier_features[:, 2].min() - margin, 
                           outlier_features[:, 2].max() + margin])
    
    plt.suptitle(f'Análise K-Means - k={n_clusters} {title_suffix}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def create_3d_visualization_dbscan(data, result, title_suffix=""):
    """
    Cria visualização 3D para resultados do DBSCAN.
    
    Args:
        data: Dados originais
        result: Resultado do DBSCAN (dict com labels, outliers, etc.)
        title_suffix: Texto adicional para o título
    
    Returns:
        matplotlib.figure.Figure: Figura criada
    """
    from src.utils.sensor_calculations import calculate_sensor_modules
    
    # Calcula módulos dos sensores
    modules = calculate_sensor_modules(data)
    features = np.column_stack([
        modules['acc_module'],
        modules['gyro_module'],
        modules['mag_module']
    ])
    
    labels = result['labels']
    outliers = result['outliers']
    n_clusters = result['n_clusters']
    
    # Cria figura com 2 subplots
    fig = plt.figure(figsize=(16, 6))
    
    # Plot 1: Clusters coloridos
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Cores para clusters (outliers em cinza)
    unique_labels = set(labels)
    colors_list = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(sorted(unique_labels), colors_list):
        if label == -1:
            # Outliers em cinza
            mask = labels == label
            ax1.scatter(
                features[mask, 0], features[mask, 1], features[mask, 2],
                c='gray', s=10, alpha=0.3, label='Noise/Outliers'
            )
        else:
            # Clusters coloridos
            mask = labels == label
            ax1.scatter(
                features[mask, 0], features[mask, 1], features[mask, 2],
                c=[color], s=20, alpha=0.6, label=f'Cluster {label}'
            )
    
    ax1.set_xlabel('Módulo Acelerómetro', fontsize=10)
    ax1.set_ylabel('Módulo Giroscópio', fontsize=10)
    ax1.set_zlabel('Módulo Magnetómetro', fontsize=10)
    ax1.set_title(f'Clusters DBSCAN ({n_clusters} clusters)', fontsize=12, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 2: Outliers destacados
    ax2 = fig.add_subplot(122, projection='3d')
    colors_outliers = np.where(outliers, 'red', 'blue')
    create_3d_scatter(
        ax2, features, colors_outliers,
        'Outliers (vermelho) vs Normais (azul)'
    )
    
    plt.suptitle(f'Análise DBSCAN {title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig
