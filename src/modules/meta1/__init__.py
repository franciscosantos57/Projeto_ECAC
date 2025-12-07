"""
Módulos da Meta 1: Engenharia de Características
Análise de outliers e extração de features
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
    'load_participant_data',
    'load_all_participants_data',
    'create_boxplot_visualization',
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
    'analyze_statistical_significance',
    'extract_features_from_windows',
    'analyze_feature_set',
    'save_feature_set',
    'load_feature_set',
    'normalize_features_zscore',
    'apply_pca',
    'analyze_variance_explained',
    'create_variance_plot',
    'print_pca_analysis',
    'print_component_contributions',
    'example_feature_compression',
    'print_compression_example',
    'compare_feature_selection_methods',
    'demonstrate_feature_extraction',
    'analyze_selection_approach'
]
