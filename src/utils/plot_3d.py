"""
Funções de visualização 3D para K-Means e DBSCAN.
Centraliza todas as funções de criação de gráficos 3D do projeto.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

from src.utils.sensor_calculations import calculate_sensor_modules


def create_3d_visualization_kmeans(data, kmeans_result, outliers_mask, n_clusters, 
                                   use_modules=True, output_dir="plots", normalized_data=None):
    """
    Cria visualização 3D dos clusters e outliers detectados pelo K-Means.
    
    IMPORTANTE: Se normalized_data for fornecido, usa os dados normalizados
    para plotar (mesma escala dos centroides). Caso contrário, usa dados originais.
    
    Args:
        data: Dados originais
        kmeans_result: Resultado do K-Means
        outliers_mask: Máscara booleana de outliers
        n_clusters: Número de clusters usado
        use_modules: True se usou módulos, False se usou valores originais
        output_dir: Diretório para guardar os gráficos
        normalized_data: Dados normalizados (mesma escala dos centroides)
    """
    # Prepara dados para visualização
    if normalized_data is not None:
        x = normalized_data[:, 0]
        y = normalized_data[:, 1]
        z = normalized_data[:, 2]
        xlabel = 'Módulo Acelerómetro (Z-Score)'
        ylabel = 'Módulo Giroscópio (Z-Score)'
        zlabel = 'Módulo Magnetómetro (Z-Score)'
        space_name = "módulos normalizados"
    elif use_modules:
        modules = calculate_sensor_modules(data)
        x = modules['acc_module']
        y = modules['gyro_module']
        z = modules['mag_module']
        xlabel = 'Módulo Acelerómetro'
        ylabel = 'Módulo Giroscópio'
        zlabel = 'Módulo Magnetómetro'
        space_name = "módulos"
    else:
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
    
    # Define cores DISTINTAS para clusters
    if n_clusters <= 3:
        color_list = ['blue', 'green', 'orange']
    elif n_clusters <= 5:
        color_list = ['blue', 'green', 'orange', 'purple', 'brown']
    elif n_clusters <= 7:
        color_list = ['blue', 'green', 'orange', 'purple', 'brown', 'salmon', 'cyan']
    else:
        cmap = plt.cm.tab20(np.linspace(0, 1, 20))
        color_indices = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19]
        color_list = [cmap[i] for i in color_indices[:n_clusters]]
    
    colors = color_list[:n_clusters]
    
    # Função auxiliar para plotar em um subplot
    def plot_3d_view(ax, elev, azim, title_suffix):
        # 1. Outliers PRIMEIRO (ficam atrás)
        if np.sum(outliers_mask) > 0:
            ax.scatter(x[outliers_mask], y[outliers_mask], z[outliers_mask],
                      c='red', label=f'Outliers ({np.sum(outliers_mask)})', 
                      alpha=0.4, s=20, marker='x', linewidths=1.5, 
                      zorder=1, depthshade=True)
        
        # 2. Pontos normais por cluster
        for k in range(n_clusters):
            cluster_mask = (labels == k) & (~outliers_mask)
            if np.sum(cluster_mask) > 0:
                ax.scatter(x[cluster_mask], y[cluster_mask], z[cluster_mask],
                          c=[colors[k]], label=f'Cluster {k+1}', 
                          alpha=0.5, s=8, edgecolors='none',
                          zorder=2, depthshade=True)
        
        # 3. Centroides - caixas amarelas com texto
        if centroids.shape[1] == 3:
            for k in range(n_clusters):
                ax.text(centroids[k, 0], centroids[k, 1], centroids[k, 2], 
                       f'C{k+1}', fontsize=10, fontweight='bold',
                       color='black', zorder=100,
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', 
                                alpha=0.9, edgecolor='black', linewidth=2))
        
        ax.set_xlabel(xlabel, fontsize=10, labelpad=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, labelpad=10, fontweight='bold')
        ax.set_zlabel(zlabel, fontsize=10, labelpad=20, fontweight='bold')
        ax.set_title(f'{title_suffix}\nOutliers: {np.sum(outliers_mask)} '
                    f'({(np.sum(outliers_mask)/len(data)*100):.2f}%)',
                    fontsize=11, fontweight='bold', pad=10)
        
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.dist = 11  # Afasta câmera para evitar cortes
        
        return ax
    
    # 4 subplots com 4 ângulos diferentes
    ax1 = fig.add_subplot(221, projection='3d')
    plot_3d_view(ax1, elev=20, azim=45, title_suffix='Ângulo 1 (Frontal)')
    
    ax2 = fig.add_subplot(222, projection='3d')
    plot_3d_view(ax2, elev=20, azim=135, title_suffix='Ângulo 2 (Lateral Direita)')
    
    ax3 = fig.add_subplot(223, projection='3d')
    plot_3d_view(ax3, elev=20, azim=225, title_suffix='Ângulo 3 (Traseira)')
    
    ax4 = fig.add_subplot(224, projection='3d')
    plot_3d_view(ax4, elev=20, azim=315, title_suffix='Ângulo 4 (Lateral Esquerda)')
    
    # Legenda única e centralizada
    legend_elements = []
    
    for k in range(n_clusters):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=colors[k], markersize=8,
                                     label=f'Cluster {k+1}'))
    
    if np.sum(outliers_mask) > 0:
        legend_elements.append(Line2D([0], [0], marker='x', color='w', 
                                     markerfacecolor='red', markeredgecolor='red',
                                     markersize=7, markeredgewidth=1.5,
                                     label=f'Outliers ({np.sum(outliers_mask)})'))
    
    legend_elements.append(Line2D([0], [0], marker='s', color='w', 
                                 markerfacecolor='yellow', markeredgecolor='black',
                                 markersize=10, markeredgewidth=2,
                                 label='Centroides'))
    
    # Título geral
    fig.suptitle(f'K-Means Clustering (k={n_clusters}) - Espaço de {space_name}',
                fontsize=15, fontweight='bold', y=0.98)
    
    # Legenda abaixo
    fig.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, 0.02), ncol=min(n_clusters + 2, 7),
              fontsize=10, framealpha=0.95, edgecolor='black', 
              fancybox=True, shadow=True)
    
    plt.subplots_adjust(top=0.93, bottom=0.10, left=0.05, right=0.95, 
                       hspace=0.35, wspace=0.20)
    
    # Guarda gráfico
    os.makedirs(output_dir, exist_ok=True)
    filename = f"kmeans_3d_k{n_clusters}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()
    
    print(f"  Gráfico 3D (4 ângulos) guardado: {filepath}")


def create_3d_visualization_kmeans_zoom(data, kmeans_result, outliers_mask, n_clusters, 
                                        use_modules=True, output_dir="plots", normalized_data=None):
    """
    Cria visualização 3D com ZOOM (4 ângulos SEM OUTLIERS).
    
    Args:
        data: Dados originais
        kmeans_result: Resultado do K-Means
        outliers_mask: Máscara booleana de outliers
        n_clusters: Número de clusters usado
        use_modules: True se usou módulos
        output_dir: Diretório para guardar os gráficos
        normalized_data: Dados normalizados
    """
    # Prepara dados
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
    
    fig = plt.figure(figsize=(20, 16))
    
    # Cores
    if n_clusters <= 3:
        color_list = ['blue', 'green', 'orange']
    elif n_clusters <= 5:
        color_list = ['blue', 'green', 'orange', 'purple', 'brown']
    elif n_clusters <= 7:
        color_list = ['blue', 'green', 'orange', 'purple', 'brown', 'salmon', 'cyan']
    else:
        cmap = plt.cm.tab20(np.linspace(0, 1, 20))
        color_indices = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19]
        color_list = [cmap[i] for i in color_indices[:n_clusters]]
    
    colors = color_list[:n_clusters]
    
    def plot_zoom_view(ax, elev, azim, title_suffix):
        # Plota apenas pontos normais
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
        ax.set_zlabel(zlabel, fontsize=10, labelpad=15, fontweight='bold')
        ax.set_title(f'{title_suffix}\nZoom nos Clusters (SEM Outliers)',
                    fontsize=11, fontweight='bold', pad=10)
        
        # Zoom automático
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
        ax.dist = 11
    
    # 4 subplots com zoom
    ax1 = fig.add_subplot(221, projection='3d')
    plot_zoom_view(ax1, elev=20, azim=45, title_suffix='Ângulo 1 (Frontal)')
    
    ax2 = fig.add_subplot(222, projection='3d')
    plot_zoom_view(ax2, elev=20, azim=135, title_suffix='Ângulo 2 (Lateral Direita)')
    
    ax3 = fig.add_subplot(223, projection='3d')
    plot_zoom_view(ax3, elev=20, azim=225, title_suffix='Ângulo 3 (Traseira)')
    
    ax4 = fig.add_subplot(224, projection='3d')
    plot_zoom_view(ax4, elev=20, azim=315, title_suffix='Ângulo 4 (Lateral Esquerda)')
    
    # Legenda
    legend_elements = []
    for k in range(n_clusters):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=colors[k], markersize=8,
                                     label=f'Cluster {k+1}'))
    
    legend_elements.append(Line2D([0], [0], marker='s', color='w', 
                                 markerfacecolor='yellow', markeredgecolor='black',
                                 markersize=10, markeredgewidth=2,
                                 label='Centroides'))
    
    fig.suptitle(f'K-Means Clustering (k={n_clusters}) - ZOOM SEM OUTLIERS\n'
                f'Melhor visualização da estrutura dos clusters',
                fontsize=15, fontweight='bold', y=0.98)
    
    fig.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, 0.02), ncol=min(n_clusters + 1, 7),
              fontsize=10, framealpha=0.95, edgecolor='black', 
              fancybox=True, shadow=True)
    
    plt.subplots_adjust(top=0.93, bottom=0.10, left=0.05, right=0.95, 
                       hspace=0.35, wspace=0.20)
    
    # Guarda gráfico zoom
    os.makedirs(output_dir, exist_ok=True)
    filename = f"kmeans_3d_k{n_clusters}_zoom.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()
    
    print(f"  Gráfico 3D ZOOM (4 ângulos sem outliers) guardado: {filepath}")


def create_3d_visualization_dbscan(data, result, title_suffix=""):
    """
    Cria visualização 3D com 4 ângulos para DBSCAN.
    
    Args:
        data: Dados originais (não usado)
        result: Resultado do DBSCAN
        title_suffix: Sufixo para título
        
    Returns:
        Figura matplotlib
    """
    # Usar features normalizadas
    modules = result['normalized_features']
    labels = result['labels']
    outliers_mask = result['outliers']
    n_clusters = result['n_clusters']
    
    x = modules[:, 0]
    y = modules[:, 1]
    z = modules[:, 2]
    
    xlabel = 'Módulo Acelerómetro (Z-Score)'
    ylabel = 'Módulo Giroscópio (Z-Score)'
    zlabel = 'Módulo Magnetómetro (Z-Score)'
    
    fig = plt.figure(figsize=(20, 16))
    
    # Cores
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
    
    def plot_3d_view(ax, elev, azim, title_suffix_view):
        # Outliers
        if np.sum(outliers_mask) > 0:
            ax.scatter(x[outliers_mask], y[outliers_mask], z[outliers_mask],
                      c='red', label=f'Outliers ({np.sum(outliers_mask)})', 
                      alpha=0.4, s=20, marker='x', linewidths=1.5, 
                      zorder=1, depthshade=True)
        
        # Clusters
        for k in range(n_clusters):
            cluster_mask = (labels == k) & (~outliers_mask)
            if np.sum(cluster_mask) > 0:
                ax.scatter(x[cluster_mask], y[cluster_mask], z[cluster_mask],
                          c=[colors[k]], label=f'Cluster {k+1}', 
                          alpha=0.5, s=8, edgecolors='none',
                          zorder=2, depthshade=True)
        
        ax.set_xlabel(xlabel, fontsize=10, labelpad=10, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10, labelpad=10, fontweight='bold')
        ax.set_zlabel(zlabel, fontsize=10, labelpad=20, fontweight='bold')
        ax.set_title(f'{title_suffix_view}\nOutliers: {np.sum(outliers_mask)} '
                    f'({(np.sum(outliers_mask)/len(modules)*100):.2f}%)',
                    fontsize=11, fontweight='bold', pad=10)
        
        ax.view_init(elev=elev, azim=azim)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.dist = 11
    
    # 4 subplots
    ax1 = fig.add_subplot(221, projection='3d')
    plot_3d_view(ax1, elev=20, azim=45, title_suffix_view='Ângulo 1 (Frontal)')
    
    ax2 = fig.add_subplot(222, projection='3d')
    plot_3d_view(ax2, elev=20, azim=135, title_suffix_view='Ângulo 2 (Lateral Direita)')
    
    ax3 = fig.add_subplot(223, projection='3d')
    plot_3d_view(ax3, elev=20, azim=225, title_suffix_view='Ângulo 3 (Traseira)')
    
    ax4 = fig.add_subplot(224, projection='3d')
    plot_3d_view(ax4, elev=20, azim=315, title_suffix_view='Ângulo 4 (Lateral Esquerda)')
    
    # Legenda
    legend_elements = []
    
    for k in range(n_clusters):
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=colors[k], markersize=8,
                                     label=f'Cluster {k+1}'))
    
    if np.sum(outliers_mask) > 0:
        legend_elements.append(Line2D([0], [0], marker='x', color='w', 
                                     markerfacecolor='red', markeredgecolor='red',
                                     markersize=7, markeredgewidth=1.5,
                                     label=f'Outliers ({np.sum(outliers_mask)})'))
    
    fig.suptitle(f'DBSCAN Clustering (eps={result["eps"]}, min_samples={result["min_samples"]})',
                fontsize=15, fontweight='bold', y=0.98)
    
    fig.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, 0.02), ncol=min(n_clusters + 1, 7),
              fontsize=10, framealpha=0.95, edgecolor='black', 
              fancybox=True, shadow=True)
    
    plt.subplots_adjust(top=0.93, bottom=0.10, left=0.05, right=0.95, 
                       hspace=0.35, wspace=0.20)
    
    return fig
