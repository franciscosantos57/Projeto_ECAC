"""
EXERCÍCIO 4.3 e 4.4: Principal Component Analysis (PCA) para Compressão de Features
Implementação de PCA com normalização Z-Score e análise de variância acumulada.

OBJETIVO:
- Reduzir dimensionalidade do espaço de features
- Determinar número de dimensões para explicar 75% da variância
- Fornecer método para obter features comprimidas
- Analisar vantagens e limitações da abordagem
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import warnings

# Suprimir warnings do NumPy e sklearn (esperados em features com escalas muito diferentes)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', module='sklearn')


def normalize_features_zscore(feature_matrix):
    """
    Normaliza o feature set usando Z-Score.
    
    Z-Score: X_norm = (X - μ) / σ
    
    Args:
        feature_matrix: Array [n_windows, n_features]
        
    Returns:
        X_normalized: Array normalizado [n_windows, n_features]
        scaler: Objeto StandardScaler com parâmetros de normalização
    """
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(feature_matrix)
    
    return X_normalized, scaler


def apply_pca(X_normalized, n_components=None):
    """
    Aplica PCA ao conjunto de features normalizado.
    
    Args:
        X_normalized: Array normalizado [n_windows, n_features]
        n_components: Número de componentes (None = todas)
        
    Returns:
        pca: Objeto PCA fitado
        X_transformed: Dados transformados [n_windows, n_components]
    """
    if n_components is None:
        n_components = X_normalized.shape[1]
    
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X_normalized)
    
    return pca, X_transformed


def analyze_variance_explained(pca):
    """
    Analisa a variância explicada por cada componente.
    
    Args:
        pca: Objeto PCA fitado
        
    Returns:
        Dict com análise de variância
    """
    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)
    
    # Encontrar número de dimensões para 75%, 80%, 90%, 95% de variância
    thresholds = [0.75, 0.80, 0.90, 0.95]
    n_components_for_threshold = {}
    
    for threshold in thresholds:
        n = np.argmax(cumulative_variance >= threshold) + 1
        n_components_for_threshold[threshold] = n
    
    return {
        'variance_explained': variance_explained,
        'cumulative_variance': cumulative_variance,
        'n_components_for_threshold': n_components_for_threshold,
        'total_variance_explained': cumulative_variance[-1]
    }


def get_compressed_features(X_normalized, scaler, n_components):
    """
    Obtém features comprimidas usando PCA.
    
    Args:
        X_normalized: Array normalizado [n_windows, n_features]
        scaler: StandardScaler usado para normalização
        n_components: Número de componentes PCA
        
    Returns:
        pca: Objeto PCA fitado
        X_compressed: Features comprimidas [n_windows, n_components]
    """
    pca, X_compressed = apply_pca(X_normalized, n_components=n_components)
    return pca, X_compressed


def get_compressed_features_for_sample(sample, pca, scaler):
    """
    Obtém features comprimidas para uma amostra específica.
    
    Args:
        sample: Array [n_features] com uma amostra
        pca: Objeto PCA previamente fitado
        scaler: StandardScaler previamente fitado
        
    Returns:
        Array [n_components] com features comprimidas
    """
    # Normalizar a amostra
    sample_normalized = scaler.transform(sample.reshape(1, -1))
    
    # Transformar com PCA
    sample_compressed = pca.transform(sample_normalized)
    
    return sample_compressed[0]


def create_variance_plot(pca, variance_info, output_dir="plots/exercicio_4.3_pca"):
    """
    Cria gráfico de variância explicada acumulada vs número de componentes.
    
    Args:
        pca: Objeto PCA fitado
        variance_info: Dicionário retornado por analyze_variance_explained()
        output_dir: Diretório para salvar o gráfico
        
    Returns:
        Caminho para o arquivo salvo
    """
    os.makedirs(output_dir, exist_ok=True)
    
    cumulative_variance = variance_info['cumulative_variance']
    n_components_75 = variance_info['n_components_for_threshold'][0.75]
    
    n_components = len(cumulative_variance)
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot de variância acumulada
    ax.plot(range(1, n_components + 1), cumulative_variance * 100, 
            'b-', linewidth=2, label='Variância Acumulada')
    
    # Linhas de referência
    ax.axhline(y=75, color='r', linestyle='--', linewidth=1.5, label='75% Variância')
    ax.axvline(x=n_components_75, color='r', linestyle='--', linewidth=1.5)
    
    # Ponto no 75%
    ax.plot(n_components_75, 75, 'ro', markersize=10, 
            label=f'{n_components_75} componentes para 75%')
    
    # Preenchimento abaixo da curva
    ax.fill_between(range(1, n_components + 1), cumulative_variance * 100, 
                    alpha=0.3, color='blue')
    
    # Configuração dos eixos
    ax.set_xlabel('Número de Componentes Principais', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variância Acumulada Explicada (%)', fontsize=12, fontweight='bold')
    ax.set_title('PCA: Variância Acumulada vs Número de Componentes', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Limites dos eixos
    ax.set_xlim(0, n_components)
    ax.set_ylim(0, 105)
    
    # Legenda no canto inferior direito - estilo consistente
    legend = ax.legend(loc='lower right', fontsize=10, framealpha=0.95,
                      edgecolor='gray', fancybox=True, shadow=False,
                      borderpad=0.7, labelspacing=0.5, handlelength=2.0)
    legend.get_frame().set_linewidth(1.2)
    
    # Adicionar texto informativo acima da legenda - exatamente o mesmo estilo
    textstr = f'Dimensionalidade Original: {pca.n_features_in_}\n'
    textstr += f'Dimensionalidade Reduzida: {n_components_75}\n'
    textstr += f'Taxa de Compressão: {100 * (1 - n_components_75/pca.n_features_in_):.1f}%'
    ax.text(0.98, 0.19, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.95, 
                     edgecolor='gray', linewidth=1.2, pad=0.7))
    
    # Guarda figura
    filepath = os.path.join(output_dir, "pca_variance_explained.png")
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return filepath


def print_pca_analysis(variance_info, n_features_original):
    """
    Imprime análise dos resultados do PCA.
    
    Args:
        variance_info: Dicionário retornado por analyze_variance_explained()
        n_features_original: Número original de features
    """
    n_components_for_threshold = variance_info['n_components_for_threshold']
    cumulative_variance = variance_info['cumulative_variance']
    
    n_75 = n_components_for_threshold[0.75]
    var_75 = cumulative_variance[n_75-1] * 100
    compression = 100 * (1 - n_75 / n_features_original)
    
    print(f"Para explicar 75% da variância: {n_75} componentes ({var_75:.2f}%)")
    print(f"Compressão: {n_features_original} → {n_75} features ({compression:.1f}%)")


def print_component_contributions(pca, n_top=5):
    """
    Imprime a contribuição das features no primeiro componente.
    
    Args:
        pca: Objeto PCA fitado
        n_top: Número de componentes a analisar
    """
    # Mostrar apenas o primeiro componente (mais importante)
    component = pca.components_[0]
    var = pca.explained_variance_ratio_[0]
    
    # Encontrar as 3 features com maior contribuição absoluta
    top_indices = np.argsort(np.abs(component))[-3:][::-1]
    
    print(f"Componente principal 1 ({var*100:.1f}% variância): features {top_indices[0]}, {top_indices[1]}, {top_indices[2]} dominam")


def example_feature_compression(feature_matrix, labels, n_components_75):
    """
    Exemplifica a compressão de features para uma amostra específica.
    
    Args:
        feature_matrix: Array [n_windows, n_features]
        labels: Array com labels das atividades
        n_components_75: Número de componentes para 75% de variância
        
    Returns:
        Dict com exemplo de compressão
    """
    # Normalizar
    X_normalized, scaler = normalize_features_zscore(feature_matrix)
    
    # Aplicar PCA
    pca, X_compressed = apply_pca(X_normalized, n_components=n_components_75)
    
    # Selecionar exemplo (primeira amostra)
    example_idx = 0
    original_features = feature_matrix[example_idx]
    normalized_features = X_normalized[example_idx]
    compressed_features = X_compressed[example_idx]
    
    activity_label = labels[example_idx]
    
    return {
        'example_idx': example_idx,
        'activity_label': activity_label,
        'original_features': original_features,
        'normalized_features': normalized_features,
        'compressed_features': compressed_features,
        'n_original': len(original_features),
        'n_compressed': len(compressed_features)
    }


def print_compression_example(example_data):
    """
    Imprime exemplo de compressão de features (simplificado).
    
    Args:
        example_data: Dict retornado por example_feature_compression()
    """
    print(f"Exemplo: amostra {example_data['example_idx']} comprimida de {example_data['n_original']} → {example_data['n_compressed']} features")
