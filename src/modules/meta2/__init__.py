"""
Módulos da Meta 2: Machine Learning e Data Augmentation
Aplicação de modelos de ML e técnicas de aumento de dados

Inclui:
- data_augmentation: SMOTE para balanceamento de dados
- embedding_extraction: Análise de embeddings extraídos via transfer learning
- data_splitting: Estratégias de split within-subject e between-subject
- dataset_scenarios: Preparação de cenários (all, PCA, ReliefF)
- knn_classifier: Implementação própria do algoritmo k-NN
- classification_metrics: Cálculo de métricas de classificação
- model_evaluation: Hyperparameter tuning e testes de hipótese
"""

from .data_augmentation import (
    analyze_dataset_balance,
    demonstrate_smote
)

from .smote_balancer import (
    balance_dataset_smote
)

from .embedding_extraction import (
    load_embeddings_dataset
)

from .data_splitting import (
    split_within_subject,
    split_between_subject,
    compare_splitting_strategies
)

from .dataset_scenarios import (
    prepare_all_scenarios
)

from .knn_classifier import (
    KNNClassifier,
    train_knn,
    predict_knn
)

from .classification_metrics import (
    confusion_matrix,
    calculate_metrics,
    print_metrics,
    evaluate_classification
)

from .model_evaluation import (
    select_best_k,
    train_and_evaluate,
    perform_multiple_splits,
    evaluate_with_multiple_splits,
    hypothesis_testing,
    print_summary_table,
    plot_average_confusion_matrix
)

from .hypothesis_visualization import (
    plot_hypothesis_tests,
    find_best_model
)

from .deployment import (
    run_classification
)

from .deployment_evaluation import (
    evaluate_deployment_accuracy
)

__all__ = [
    'analyze_dataset_balance',
    'demonstrate_smote',
    'balance_dataset_smote',
    'load_embeddings_dataset',
    'split_within_subject',
    'split_between_subject',
    'compare_splitting_strategies',
    'prepare_all_scenarios',
    'print_scenarios_summary',
    'KNNClassifier',
    'train_knn',
    'predict_knn',
    'confusion_matrix',
    'calculate_metrics',
    'print_metrics',
    'evaluate_classification',
    'select_best_k',
    'train_and_evaluate',
    'compare_confusion_matrices',
    'perform_multiple_splits',
    'evaluate_with_multiple_splits',
    'hypothesis_testing',
    'print_summary_table',
    'plot_confusion_matrix',
    'plot_hypothesis_tests',
    'find_best_model',
    'run_classification',
    'evaluate_deployment_accuracy'
]
