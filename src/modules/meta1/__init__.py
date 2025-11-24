"""
Módulos da Meta 1: Engenharia de Características
Análise de outliers e extração de features

Inclui:
- data_loader: Carregamento de dados de sensores
- boxplot_visualization: Visualização de boxplots dos módulos dos sensores
- outlier_density_analysis: Análise de densidade de outliers
- zscore_outlier_detection: Detecção de outliers usando Z-Score
- kmeans_outlier_detection: Detecção de outliers usando K-Means
- dbscan_outlier_detection: Detecção de outliers usando DBSCAN
- statistical_significance: Testes de significância estatística
- feature_extraction: Extração de features temporais e espectrais
- feature_analysis: Análise e salvamento de features
- pca_analysis: Análise de componentes principais
- feature_comparison: Comparação entre métodos de seleção de features
"""

from .data_loader import (
    load_participant_data,
    load_all_participants_data
)

from .boxplot_visualization import (
    create_boxplot_visualization
)

from .outlier_density_analysis import (
    calculate_outlier_density,
    analyze_outlier_patterns
)

from .zscore_outlier_detection import (
    detect_outliers_zscore,
    create_zscore_plots,
    compare_methods
)

from .kmeans_outlier_detection import (
    detect_outliers_kmeans,
    analyze_kmeans_outliers,
    compare_with_zscore
)

from .dbscan_outlier_detection import (
    detect_outliers_dbscan,
    analyze_dbscan_outliers,
    summarize_dbscan_analysis
)

from .statistical_significance import (
    analyze_statistical_significance
)

from .feature_extraction import (
    extract_features_from_windows
)

from .feature_analysis import (
    analyze_feature_set,
    save_feature_set,
    load_feature_set
)

from .pca_analysis import (
    normalize_features_zscore,
    apply_pca,
    analyze_variance_explained,
    create_variance_plot,
    print_pca_analysis,
    print_component_contributions,
    example_feature_compression,
    print_compression_example
)

from .feature_comparison import (
    compare_feature_selection_methods,
    demonstrate_feature_extraction,
    analyze_selection_approach
)

__all__ = [
    # Data loading
    'load_participant_data',
    'load_all_participants_data',
    
    # Visualization
    'create_boxplot_visualization',
    
    # Outlier detection
    'calculate_outlier_density',
    'analyze_outlier_patterns',
    'detect_outliers_zscore',
    'create_zscore_plots',
    'compare_methods',
    'detect_outliers_kmeans',
    'analyze_kmeans_outliers',
    'compare_with_zscore',
    'detect_outliers_dbscan',
    'analyze_dbscan_outliers',
    'summarize_dbscan_analysis',
    
    # Statistical analysis
    'analyze_statistical_significance',
    
    # Feature extraction and analysis
    'extract_features_from_windows',
    'analyze_feature_set',
    'save_feature_set',
    'load_feature_set',
    
    # PCA
    'normalize_features_zscore',
    'apply_pca',
    'analyze_variance_explained',
    'create_variance_plot',
    'print_pca_analysis',
    'print_component_contributions',
    'example_feature_compression',
    'print_compression_example',
    
    # Feature comparison
    'compare_feature_selection_methods',
    'demonstrate_feature_extraction',
    'analyze_selection_approach'
]
