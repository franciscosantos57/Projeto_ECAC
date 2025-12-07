"""
Avaliação de Accuracy do Deployment
Testa a função run_classification múltiplas vezes para calcular accuracy média.
"""

import numpy as np
from tqdm import tqdm

from src.modules.meta2.deployment import run_classification


def evaluate_deployment_accuracy(model_name, distributions_within, distributions_between,
                                 X_features, X_embeddings, y_labels, participant_ids,
                                 n_iterations):
    """
    Avalia a accuracy de um modelo executando múltiplas classificações.
    
    Args:
        model_name (str): Nome do modelo - formato: 'within_features_all', 'between_embeddings_pca', etc.
        distributions_within (dict): Resultados do exercício 5.3 (within-subject)
        distributions_between (dict): Resultados do exercício 5.3 (between-subject)
        X_features (np.ndarray): Dataset completo de features
        X_embeddings (np.ndarray): Dataset completo de embeddings
        y_labels (np.ndarray): Labels do dataset
        participant_ids (np.ndarray): IDs dos participantes para cada amostra
        n_iterations (int): Número de classificações a executar
    
    Returns:
        dict: Dicionário com estatísticas de avaliação contendo:
            - accuracy: Accuracy geral (proporção de acertos)
            - n_correct: Número de classificações corretas
            - n_total: Número total de classificações
    """
    
    # Armazena contadores
    n_correct = 0
    n_total = 0
    
    # TREINA MODELO UMA ÚNICA VEZ (primeira iteração)
    trained_model = None
    
    # Executa classificações com barra de progresso
    for i in tqdm(range(n_iterations), desc="Classificando", ncols=80):
        try:
            result = run_classification(
                model_name,
                distributions_within,
                distributions_between,
                X_features,
                X_embeddings,
                y_labels,
                participant_ids,
                trained_model=trained_model
            )
            
            # Guarda modelo treinado para reutilizar nas próximas iterações
            if i == 0:
                trained_model = result['trained_model']
            
            # Verifica se classificação foi correta
            if result['predicted_label'] == result['true_label']:
                n_correct += 1
            
            n_total += 1
            
        except Exception:
            continue
    
    # Calcula accuracy
    accuracy = n_correct / n_total if n_total > 0 else 0.0
    
    return {
        'accuracy': accuracy,
        'n_correct': n_correct,
        'n_total': n_total
    }

