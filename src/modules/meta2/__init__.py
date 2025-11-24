"""
Módulos da Meta 2: Machine Learning e Data Augmentation
Aplicação de modelos de ML e técnicas de aumento de dados

Inclui:
- data_augmentation: SMOTE para balanceamento de dados
- embedding_extraction: Análise de embeddings extraídos via transfer learning
"""

from .data_augmentation import (
    analyze_dataset_balance,
    demonstrate_smote
)

from .embedding_extraction import (
    load_embeddings_dataset
)

__all__ = [
    'analyze_dataset_balance',
    'demonstrate_smote',
    'load_embeddings_dataset'
]
