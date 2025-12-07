"""
EXERCÍCIO 6: Deployment - Classificação de Dados de Sensores
Função que recebe modelo escolhido e retorna a classificação.
"""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from src.modules.meta1.feature_extraction import extract_features_from_windows
from src.modules.meta1.pca_analysis import apply_pca
from src.modules.meta1.relieff_selection import calculate_relieff_score
from src.utils.embeddings_extractor import load_model, resample_to_30hz_5s
from src.modules.meta1.data_loader import load_participant_data
from src.modules.meta2.smote_balancer import balance_dataset_smote


def run_classification(model_name, distributions_within, distributions_between,
                      X_features, X_embeddings, y_labels):
    """
    Executa classificação completa: seleciona array aleatório, aplica variação realista, e remove-o do dataset de treino.
    
    Pipeline:
        1. Seleciona participante e atividade aleatórios
        2. Extrai janela de 256 amostras
        3. Aplica variação realista (scaling + noise)
        4. Identifica o índice correspondente no dataset
        5. Remove essa amostra do dataset de treino
        6. Busca melhor k do modelo escolhido (do exercício 5)
        7. Aplica SMOTE para balancear dados de treino
        8. Treina modelo com dados balanceados
        9. Extrai features/embeddings da janela de teste (com variação)
        10. Normalização (Z-Score)
        11. Redução dimensional (PCA ou ReliefF, se aplicável)
        12. Classificação (k-NN)
        13. Verifica se classificação está correta
    
    Args:
        model_name (str): Nome do modelo - formato: 'within_features_all', 'between_embeddings_pca', etc.
                         Padrão: '{scenario}_{tipo}_{reducao}'
        distributions_within (dict): Resultados do exercício 5.3 (within-subject)
        distributions_between (dict): Resultados do exercício 5.3 (between-subject)
        X_features (np.ndarray): Dataset completo de features
        X_embeddings (np.ndarray): Dataset completo de embeddings
        y_labels (np.ndarray): Labels do dataset
    
    Returns:
        dict: Dicionário com resultados da classificação contendo:
            - best_k: Valor k otimizado
            - participant: ID do participante selecionado
            - true_label: Label verdadeira da atividade
            - n_train_samples: Número de amostras de treino (após remover teste)
            - n_balanced_samples: Número de amostras após SMOTE
            - test_array_shape: Shape do array de teste
            - scale_factor: Fator de escala aplicado na variação
            - predicted_label: Label predita pelo modelo
    """
    
    # Extrai scenario (within/between), tipo (features/embeddings) e redução (all/pca/relieff)
    parts = model_name.split('_')
    if len(parts) < 3:
        raise ValueError(f"Formato inválido. Use: 'within_features_all' ou 'between_embeddings_pca'")
    
    scenario_type = parts[0]  # within ou between
    model_type = parts[1]     # features ou embeddings
    reduction = parts[2]      # all, pca, relieff
    
    # Seleciona distribution correto
    distributions = distributions_within if scenario_type == 'within' else distributions_between
    
    # Busca melhor k do exercício 5.3
    best_k = distributions[model_type][reduction]['best_k']
    
    # Seleciona dataset correto
    X_full = X_embeddings if model_type == 'embeddings' else X_features
    
    # Garante aleatoriedade verdadeira (não usa seed fixa)
    np.random.seed(None)
    
    # Seleciona participante e atividade aleatórios
    random_participant = np.random.randint(0, 15)
    random_activity = np.random.randint(1, 8)  # Atividades 1-7 (como no dataset filtrado)
    
    # Carrega dados do participante de teste
    participant_data = load_participant_data(random_participant)
    
    # Filtra apenas a atividade escolhida
    activity_data = participant_data[participant_data[:, 11] == random_activity]
    
    if len(activity_data) < 256:
        raise ValueError(f"Atividade {random_activity} do participante {random_participant} tem menos de 256 amostras")
    
    # Seleciona janela aleatória de 256 amostras
    max_start = len(activity_data) - 256
    start_idx = np.random.randint(0, max_start + 1)
    test_array_original = activity_data[start_idx:start_idx + 256, 1:10]  # Colunas 1-9 (sensores)
    
    # Aplica variação realista (scaling + noise)
    test_array, scale_factor = generate_realistic_variation(test_array_original)
    
    true_label = int(random_activity)  # Atividade 1-7
    test_array_shape = test_array.shape
    
    # Extração de Features ou Embeddings da amostra de teste (com variação aplicada)
    if model_type == 'embeddings':
        X_test = _extract_embeddings(test_array)
    else:
        X_test = _extract_features(test_array)
    
    # IDENTIFICAR ÍNDICE MAIS PRÓXIMO NO DATASET PARA REMOVER
    # Calcula distância euclidiana entre X_test e todas as amostras do dataset
    distances = np.linalg.norm(X_full - X_test, axis=1)
    test_idx = np.argmin(distances)
    
    # Remove a amostra de teste do dataset de treino
    train_mask = np.ones(len(X_full), dtype=bool)
    train_mask[test_idx] = False
    
    X_train = X_full[train_mask]
    y_train = y_labels[train_mask]
    n_train_samples = len(y_train)
    
    # ETAPA 1: Normaliza dados de treino
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_train)
    
    # ETAPA 2: Aplica redução dimensional (se aplicável)
    pca_model = None
    relieff_indices = None
    
    if reduction == 'pca':
        # Calcula PCA nos dados normalizados
        pca_full, _ = apply_pca(X_normalized, n_components=None)
        cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = np.argmax(cumsum_var >= 0.90) + 1
        
        # Aplica PCA com n_components otimizado
        pca_model, X_normalized = apply_pca(X_normalized, n_components=n_components)
        
    elif reduction == 'relieff':
        # Calcula ReliefF nos dados normalizados
        relieff_scores = calculate_relieff_score(
            X_normalized, y_train, n_neighbors=10,
            n_samples=min(100, len(X_normalized))
        )
        relieff_indices = np.argsort(relieff_scores)[::-1][:15]
        X_normalized = X_normalized[:, relieff_indices]
    
    # ETAPA 3: Aplica SMOTE nos dados normalizados e reduzidos
    X_balanced, y_balanced = balance_dataset_smote(X_normalized, y_train, n_neighbors=5, verbose=False)
    n_balanced_samples = len(y_balanced)
    
    # ETAPA 4: Treina k-NN
    knn_model = KNeighborsClassifier(n_neighbors=best_k, metric='manhattan', weights='distance')
    knn_model.fit(X_balanced, y_balanced)
    
    # ETAPA 5: Normalização da amostra de teste
    X_normalized_test = scaler.transform(X_test.reshape(1, -1))
    
    # ETAPA 6: Aplica a mesma redução dimensional
    if pca_model is not None:
        X_processed_test = pca_model.transform(X_normalized_test)
    elif relieff_indices is not None:
        X_processed_test = X_normalized_test[:, relieff_indices]
    else:
        X_processed_test = X_normalized_test
    
    # ETAPA 7: Classificação
    prediction = knn_model.predict(X_processed_test)
    predicted_label = int(prediction[0])
    
    # Retorna dicionário com todos os resultados
    return {
        'best_k': best_k,
        'participant': random_participant,
        'true_label': true_label,
        'n_train_samples': n_train_samples,
        'n_balanced_samples': n_balanced_samples,
        'test_array_shape': test_array_shape,
        'scale_factor': scale_factor,
        'predicted_label': predicted_label
    }


def _extract_features(sensor_array):
    """
    Extrai features temporais/espectrais de um array de sensores.
    
    Args:
        sensor_array (np.ndarray): Array de sensores com shape (256, 9)
            Colunas: [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, mag_x, mag_y, mag_z]
    
    Returns:
        np.ndarray: Array com 174 features extraídas
    """
    # extract_features_from_windows espera coluna 0 = device_id
    # Adiciona coluna fake de device_id = 0
    device_col = np.zeros((sensor_array.shape[0], 1))
    data_with_device = np.hstack([device_col, sensor_array])
    
    # Formato correto esperado por extract_features_from_windows
    window = [{
        'data': data_with_device,
        'activity': 0,
        'device': 0,
        'participant_id': 0,
        'start_idx': 0,
        'end_idx': 256,
        'is_valid': True
    }]
    
    feature_matrix, _, _, _, _ = extract_features_from_windows(
        window, sampling_rate=50, extract_embeddings=False
    )
    return feature_matrix[0]


def _extract_embeddings(sensor_array):
    """
    Extrai embeddings de um array de sensores usando HARNet5.
    
    Args:
        sensor_array (np.ndarray): Array de sensores com shape (256, 9)
            Colunas: [acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, mag_x, mag_y, mag_z]
    
    Returns:
        np.ndarray: Array com 512 embeddings
    """
    feature_encoder = load_model()
    # sensor_array já tem apenas sensores (colunas 0-8), sem device_id
    # Acelerômetro está nas colunas 0, 1, 2
    acc_xyz = sensor_array[:, 0:3]
    acc_resampled, _ = resample_to_30hz_5s(acc_xyz, 51.5)
    x_input = torch.from_numpy(acc_resampled.T).float().unsqueeze(0)
    
    with torch.no_grad():
        embeddings = feature_encoder(x_input).cpu().numpy()
    
    return embeddings[0]


def generate_realistic_variation(sensor_array):
    """
    Gera uma variação realista de uma janela de sensores.
    Combina Scaling (variação de esforço) com Noise (ruído do sensor).
    """
    # Simula variação de força/intensidade (cria um fator de intensidade de 0.9 a 1.1)
    scale_factor = np.random.uniform(0.9, 1.1)
    scaled_array = sensor_array * scale_factor
    
    # Simula imperfeição do sensor (adiciona ruído gaussiano com desvio padrão de 10% do desvio padrão original)
    noise_std = np.std(sensor_array, axis=0)
    noise = np.random.normal(0, noise_std * 0.1, sensor_array.shape)
    
    # Combina os dois
    new_array = scaled_array + noise
    
    return new_array, scale_factor