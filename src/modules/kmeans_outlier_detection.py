"""
EXERCÍCIO 3.6 e 3.7: Implementação e uso de K-Means para deteção de outliers.
Agrupa dados em clusters e identifica outliers baseado em distância aos centroides.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sem interface gráfica
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

from src.utils.sensor_calculations import calculate_sensor_modules, zscore_normalization


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
        # Usa espaço dos módulos (3D)
        modules = calculate_sensor_modules(data)
        clustering_data = np.column_stack([
            modules['acc_module'],
            modules['gyro_module'],
            modules['mag_module']
        ])
        space_name = "módulos"
    else:
        # Usa espaço original dos eixos x, y, z (9D)
        clustering_data = data[:, 1:10]  # Colunas 2-10: acc_x até mag_z
        space_name = "valores originais"
    
    print(f"Usando espaço de {space_name}")
    print(f"Dimensão dos dados: {clustering_data.shape}")
    
    # NORMALIZAÇÃO Z-SCORE
    print("Normalizando dados com Z-Score...")
    clustering_data_normalized = zscore_normalization(clustering_data)
    print(f"  ✓ Dados normalizados: média ≈ 0, desvio padrão ≈ 1")
    
    results = {}
    
    for n_clusters in n_clusters_list:
        print(f"\nExecutando K-Means com k={n_clusters}...")
        
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
        
        print(f"  • Clusters formados: {n_clusters}")
        print(f"  • Iterações até convergência: {kmeans_result['n_iter']}")
        print(f"  • Inércia: {kmeans_result['inertia']:.2f}")
        print(f"  • Distância média aos centroides: {mean_distance:.3f}")
        print(f"  • Distância mediana aos centroides: {median_distance:.3f}")
        print(f"  • Q1 (percentil 25): {Q1:.3f}")
        print(f"  • Q3 (percentil 75): {Q3:.3f}")
        print(f"  • IQR (Q3 - Q1): {IQR:.3f}")
        print(f"  • Threshold para outliers (Q3 + 1.5×IQR): {threshold:.3f}")
        print(f"  • Outliers detectados: {n_outliers} ({outlier_percentage:.2f}%)")
        
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
            'normalized_data': clustering_data_normalized  # Adiciona dados normalizados
        }
    
    return results

def create_3d_visualization(data, kmeans_result, outliers_mask, n_clusters, 
                            use_modules=True, output_dir="plots", normalized_data=None):
    """
    Cria visualização 3D dos clusters e outliers detectados pelo K-Means.
    
    IMPORTANTE: Se normalized_data for fornecido, usa os dados normalizados
    para plotar (mesma escala dos centroides). Caso contrário, usa dados originais.
    
    Args:
        data (numpy.ndarray): Dados originais
        kmeans_result (dict): Resultado do K-Means
        outliers_mask (numpy.ndarray): Máscara booleana de outliers
        n_clusters (int): Número de clusters usado
        use_modules (bool): True se usou módulos, False se usou valores originais
        output_dir (str): Diretório para salvar os gráficos
        normalized_data (numpy.ndarray): Dados normalizados (mesma escala dos centroides)
    """
    # Prepara dados para visualização
    # Se normalized_data foi fornecido, usa ele (dados em escala Z-Score)
    # Caso contrário, calcula módulos dos dados originais
    if normalized_data is not None:
        # Usa dados normalizados (mesma escala dos centroides)
        x = normalized_data[:, 0]
        y = normalized_data[:, 1]
        z = normalized_data[:, 2]
        xlabel = 'Módulo Acelerómetro (Z-Score)'
        ylabel = 'Módulo Giroscópio (Z-Score)'
        zlabel = 'Módulo Magnetómetro (Z-Score)'
        space_name = "módulos normalizados"
    elif use_modules:
        # Usa dados originais (não normalizados)
        modules = calculate_sensor_modules(data)
        x = modules['acc_module']
        y = modules['gyro_module']
        z = modules['mag_module']
        xlabel = 'Módulo Acelerómetro'
        ylabel = 'Módulo Giroscópio'
        zlabel = 'Módulo Magnetómetro'
        space_name = "módulos"
    else:
        # Para visualização 3D do espaço 9D, usa PCA ou seleciona 3 dimensões principais
        # Por simplicidade, vamos usar as 3 primeiras componentes (acc_x, acc_y, acc_z)
        x = data[:, 1]  # acc_x
        y = data[:, 2]  # acc_y
        z = data[:, 3]  # acc_z
        xlabel = 'Aceleração X'
        ylabel = 'Aceleração Y'
        zlabel = 'Aceleração Z'
        space_name = "valores originais"
    
    labels = kmeans_result['labels']
    centroids = kmeans_result['centroids']
    
    # Cria figura com 4 subplots (4 ângulos diferentes)
    fig = plt.figure(figsize=(20, 16))
    
    # Define cores DISTINTAS para clusters (sem vermelho, sem cores parecidas)
    # Usa colormap qualitativa com cores bem diferentes
    if n_clusters <= 3:
        color_list = ['blue', 'green', 'orange']
    elif n_clusters <= 5:
        color_list = ['blue', 'green', 'orange', 'purple', 'brown']
    elif n_clusters <= 7:
        color_list = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
    else:
        # Para mais clusters, usa tab20 (20 cores distintas) excluindo vermelhos
        cmap = plt.cm.tab20(np.linspace(0, 1, 20))
        # Remove cores vermelhas/laranjas próximas (índices 6, 7, 12, 13)
        color_indices = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19]
        color_list = [cmap[i] for i in color_indices[:n_clusters]]
    
    colors = color_list[:n_clusters]
    
    # Função auxiliar para plotar em um subplot
    def plot_3d_view(ax, elev, azim, title_suffix):
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
        
        # 3. Centroides - APENAS caixas amarelas com texto (SEM estrelas)
        # Se normalized_data foi fornecido, os centroides já estão na escala correta
        # Se não, só plota se use_modules for True e centroides forem 3D
        if centroids.shape[1] == 3:
            # Adiciona apenas as caixas de texto nos centroides (sem marcadores)
            for k in range(n_clusters):
                ax.text(centroids[k, 0], centroids[k, 1], centroids[k, 2], 
                       f'C{k+1}', fontsize=10, fontweight='bold',
                       color='black', zorder=100,
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                                alpha=0.9, edgecolor='black', linewidth=2))
        
        ax.set_xlabel(xlabel, fontsize=10, labelpad=8, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, labelpad=8, fontweight='bold')
        ax.set_zlabel(zlabel, fontsize=10, labelpad=8, fontweight='bold')
        ax.set_title(f'{title_suffix}\nOutliers: {np.sum(outliers_mask)} '
                    f'({(np.sum(outliers_mask)/len(data)*100):.2f}%)',
                    fontsize=11, fontweight='bold', pad=10)
        
        # Ajusta visualização com ângulo específico
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        return ax
    
    # 4 subplots com 4 ângulos diferentes (2x2 grid)
    # Subplot 1: Vista Frontal (elev=20, azim=45)
    ax1 = fig.add_subplot(221, projection='3d')
    plot_3d_view(ax1, elev=20, azim=45, title_suffix='Ângulo 1 (Frontal)')
    
    # Subplot 2: Vista Lateral Direita (elev=20, azim=135)
    ax2 = fig.add_subplot(222, projection='3d')
    plot_3d_view(ax2, elev=20, azim=135, title_suffix='Ângulo 2 (Lateral Direita)')
    
    # Subplot 3: Vista Traseira (elev=20, azim=225)
    ax3 = fig.add_subplot(223, projection='3d')
    plot_3d_view(ax3, elev=20, azim=225, title_suffix='Ângulo 3 (Traseira)')
    
    # Subplot 4: Vista Lateral Esquerda (VOLTOU AO NORMAL - SEM ZOOM)
    ax4 = fig.add_subplot(224, projection='3d')
    plot_3d_view(ax4, elev=20, azim=315, title_suffix='Ângulo 4 (Lateral Esquerda)')
    
    # Legenda única e centralizada abaixo dos gráficos (para não tapar título)
    # Cria handles customizados SEM estrela (apenas caixas amarelas para centroides)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
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
    
    # Adiciona centroides como QUADRADO amarelo (não estrela)
    legend_elements.append(Line2D([0], [0], marker='s', color='w', 
                                 markerfacecolor='yellow', markeredgecolor='black',
                                 markersize=10, markeredgewidth=2,
                                 label='Centroides'))
    
    # Título geral da figura NO TOPO
    fig.suptitle(f'K-Means Clustering (k={n_clusters}) - Espaço de {space_name}',
                fontsize=15, fontweight='bold', y=0.98)
    
    # Posiciona legenda ABAIXO dos gráficos para não tapar o título
    fig.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, 0.02), ncol=min(n_clusters + 2, 7),
              fontsize=10, framealpha=0.95, edgecolor='black', 
              fancybox=True, shadow=True)
    
    # Ajusta layout: mais espaço em cima para título e embaixo para legenda
    plt.subplots_adjust(top=0.94, bottom=0.08, left=0.05, right=0.95, 
                       hspace=0.25, wspace=0.25)
    
    # Salva o gráfico
    os.makedirs(output_dir, exist_ok=True)
    filename = f"kmeans_3d_k{n_clusters}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Gráfico 3D (4 ângulos) salvo: {filepath}")


def create_3d_visualization_zoom(data, kmeans_result, outliers_mask, n_clusters, 
                                   use_modules=True, output_dir="plots", normalized_data=None):
    """
    Cria visualização 3D com ZOOM (4 ângulos SEM OUTLIERS) em arquivo SEPARADO.
    
    Esta função cria um arquivo adicional mostrando APENAS os clusters (sem outliers)
    com zoom automático para melhor visualização da estrutura dos clusters.
    
    Args:
        data (numpy.ndarray): Dados originais
        kmeans_result (dict): Resultado do K-Means
        outliers_mask (numpy.ndarray): Máscara booleana de outliers
        n_clusters (int): Número de clusters usado
        use_modules (bool): True se usou módulos, False se usou valores originais
        output_dir (str): Diretório para salvar os gráficos
        normalized_data (numpy.ndarray): Dados normalizados (mesma escala dos centroides)
    """
    # Prepara dados para visualização (mesmo código da função original)
    if normalized_data is not None:
        x = normalized_data[:, 0]
        y = normalized_data[:, 1]
        z = normalized_data[:, 2]
        xlabel = 'Módulo Acelerómetro (Z-Score)'
        ylabel = 'Módulo Giroscópio (Z-Score)'
        zlabel = 'Módulo Magnetómetro (Z-Score)'
    elif use_modules:
        modules = calculate_sensor_modules(data)
        x = modules['acc_module']
        y = modules['gyro_module']
        z = modules['mag_module']
        xlabel = 'Módulo Acelerómetro'
        ylabel = 'Módulo Giroscópio'
        zlabel = 'Módulo Magnetómetro'
    else:
        x = data[:, 1]
        y = data[:, 2]
        z = data[:, 3]
        xlabel = 'Aceleração X'
        ylabel = 'Aceleração Y'
        zlabel = 'Aceleração Z'
    
    labels = kmeans_result['labels']
    centroids = kmeans_result['centroids']
    
    # Cria figura com 4 subplots (TODOS COM ZOOM, SEM OUTLIERS)
    fig = plt.figure(figsize=(20, 16))
    
    # Define cores (mesmo esquema)
    if n_clusters <= 3:
        color_list = ['blue', 'green', 'orange']
    elif n_clusters <= 5:
        color_list = ['blue', 'green', 'orange', 'purple', 'brown']
    elif n_clusters <= 7:
        color_list = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
    else:
        cmap = plt.cm.tab20(np.linspace(0, 1, 20))
        color_indices = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19]
        color_list = [cmap[i] for i in color_indices[:n_clusters]]
    
    colors = color_list[:n_clusters]
    
    # Função para plotar ZOOM (sem outliers)
    def plot_zoom_view(ax, elev, azim, title_suffix):
        # Plota APENAS pontos normais (sem outliers)
        for k in range(n_clusters):
            cluster_mask = (labels == k) & (~outliers_mask)
            if np.sum(cluster_mask) > 0:
                ax.scatter(x[cluster_mask], y[cluster_mask], z[cluster_mask],
                          c=[colors[k]], label=f'Cluster {k+1}', 
                          alpha=0.6, s=15, edgecolors='none',
                          zorder=2, depthshade=True)
        
        # Centroides
        if centroids.shape[1] == 3:
            for k in range(n_clusters):
                ax.text(centroids[k, 0], centroids[k, 1], centroids[k, 2], 
                       f'C{k+1}', fontsize=10, fontweight='bold',
                       color='black', zorder=100,
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                                alpha=0.9, edgecolor='black', linewidth=2))
        
        ax.set_xlabel(xlabel, fontsize=10, labelpad=8, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, labelpad=8, fontweight='bold')
        ax.set_zlabel(zlabel, fontsize=10, labelpad=8, fontweight='bold')
        ax.set_title(f'{title_suffix}\nZoom nos Clusters (SEM Outliers)',
                    fontsize=11, fontweight='bold', pad=10)
        
        # Ajusta limites para zoom (sem outliers)
        non_outlier_mask = ~outliers_mask
        if np.sum(non_outlier_mask) > 0:
            x_data = x[non_outlier_mask]
            y_data = y[non_outlier_mask]
            z_data = z[non_outlier_mask]
            
            x_margin = (np.max(x_data) - np.min(x_data)) * 0.1
            y_margin = (np.max(y_data) - np.min(y_data)) * 0.1
            z_margin = (np.max(z_data) - np.min(z_data)) * 0.1
            
            ax.set_xlim([np.min(x_data) - x_margin, np.max(x_data) + x_margin])
            ax.set_ylim([np.min(y_data) - y_margin, np.max(y_data) + y_margin])
            ax.set_zlim([np.min(z_data) - z_margin, np.max(z_data) + z_margin])
        
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # 4 subplots com ZOOM (todos os ângulos SEM outliers)
    ax1 = fig.add_subplot(221, projection='3d')
    plot_zoom_view(ax1, elev=20, azim=45, title_suffix='Ângulo 1 (Frontal)')
    
    ax2 = fig.add_subplot(222, projection='3d')
    plot_zoom_view(ax2, elev=20, azim=135, title_suffix='Ângulo 2 (Lateral Direita)')
    
    ax3 = fig.add_subplot(223, projection='3d')
    plot_zoom_view(ax3, elev=20, azim=225, title_suffix='Ângulo 3 (Traseira)')
    
    ax4 = fig.add_subplot(224, projection='3d')
    plot_zoom_view(ax4, elev=20, azim=315, title_suffix='Ângulo 4 (Lateral Esquerda)')
    
    # Legenda
    from matplotlib.lines import Line2D
    
    legend_elements = []
    for k in range(n_clusters):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=colors[k], markersize=8,
                                     label=f'Cluster {k+1}'))
    
    # Título geral
    fig.suptitle(f'K-Means Clustering (k={n_clusters}) - ZOOM SEM OUTLIERS\n'
                f'Melhor visualização da estrutura dos clusters',
                fontsize=15, fontweight='bold', y=0.98)
    
    # Legenda centralizada
    fig.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, 0.02), ncol=min(n_clusters, 7),
              fontsize=10, framealpha=0.95, edgecolor='black', 
              fancybox=True, shadow=True)
    
    plt.subplots_adjust(top=0.94, bottom=0.08, left=0.05, right=0.95, 
                       hspace=0.25, wspace=0.25)
    
    # Salva arquivo SEPARADO com sufixo "_zoom"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"kmeans_3d_k{n_clusters}_zoom.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Gráfico 3D ZOOM (4 ângulos sem outliers) salvo: {filepath}")


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
    print("ANÁLISE DE OUTLIERS COM K-MEANS CLUSTERING")
    print("="*60)
    
    # Usar apenas 1/5 dos dados (20% do dataset)
    sample_size = len(data) // 5
    np.random.seed(42)
    indices = np.random.choice(len(data), sample_size, replace=False)
    data_sample = data[indices]
    print(f"\nUsando amostra de {sample_size:,} pontos (1/5 de {len(data):,} totais)")
    
    # Deteta outliers com K-Means
    results = detect_outliers_kmeans(data_sample, n_clusters_list=cluster_range, 
                                    use_modules=use_modules)
    
    # Cria visualizações 3D
    if create_plots:
        print("\nCriando visualizações 3D...")
        for n_clusters in cluster_range:
            # Gráfico normal (4 ângulos com outliers)
            create_3d_visualization(
                data_sample, 
                results[n_clusters]['kmeans_result'],
                results[n_clusters]['outliers_mask'],
                n_clusters,
                use_modules=use_modules,
                normalized_data=results[n_clusters]['normalized_data'],
                output_dir=output_dir_normal
            )
            
            # Gráfico ZOOM (4 ângulos SEM outliers) - arquivo separado
            create_3d_visualization_zoom(
                data_sample, 
                results[n_clusters]['kmeans_result'],
                results[n_clusters]['outliers_mask'],
                n_clusters,
                use_modules=use_modules,
                normalized_data=results[n_clusters]['normalized_data'],
                output_dir=output_dir_zoom
            )
    
    # Resumo comparativo
    print("\n" + "="*60)
    print("RESUMO COMPARATIVO (Método IQR)")
    print("="*60)
    print(f"{'k':<5} {'Inércia':<14} {'Mediana':<10} {'IQR':<10} {'Threshold':<12} {'Outliers':<10} {'%':<8}")
    print("-"*60)
    
    for n_clusters in sorted(results.keys()):
        r = results[n_clusters]
        print(f"{n_clusters:<5} {r['kmeans_result']['inertia']:<14.2f} "
              f"{r['median_distance']:<10.3f} {r['IQR']:<10.3f} {r['threshold']:<12.3f} "
              f"{r['n_outliers']:<10} {r['outlier_percentage']:<8.2f}")
    
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
    
    print("\n" + "="*60)
    print(f"COMPARAÇÃO: K-MEANS vs Z-SCORE (k={k_zscore})")
    print("="*60)
    print(f"Z-Score: {n_zscore} outliers ({n_zscore/len(data_to_use)*100:.2f}%)")
    print("-"*60)
    
    for n_clusters in sorted(actual_results.keys()):
        kmeans_outliers = actual_results[n_clusters]['outliers_mask']
        n_kmeans = np.sum(kmeans_outliers)
        
        # Sobreposição
        overlap = np.sum(kmeans_outliers & zscore_outliers)
        overlap_pct = (overlap / n_zscore * 100) if n_zscore > 0 else 0
        
        print(f"K-Means (k={n_clusters}): {n_kmeans} outliers ({n_kmeans/len(data_to_use)*100:.2f}%)")
        print(f"  → Sobreposição com Z-Score: {overlap} ({overlap_pct:.1f}%)")
