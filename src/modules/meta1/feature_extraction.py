"""
EXERCÍCIO 4.2: Extração de Features Temporais e Espectrais
Baseado no artigo: Zhang & Sawchuk - "A feature selection based framework 
for human activity recognition using wearable multimodal sensors"

Features implementadas:
- Tabela 1 - Temporais (14 features): Mean, Median, Std, Variance, Min, Max, Range, 
  RMS, MAD (Averaged derivatives), IQR, Skewness, Kurtosis, ZCR, MCR
- Features espectrais (4): Dominant Frequency, Spectral Energy, Spectral Entropy, 
  Spectral Centroid
- Features adicionais do paper (12): AI, VI (Movement Intensity), SMA, 
  EVA1/EVA2/EVA3 (eigenvalues), CAGH, AVH, AVG, ARATG, AAE, ARE
  
Total: 9 eixos × 18 features + 12 multi-sensor = 174 features

INTEGRAÇÃO META 2 - EXERCÍCIO 2.1:
Durante a extração de features, também extrai embeddings (512-dim) usando
o modelo harnet5 pré-treinado. Os embeddings são extraídos dos mesmos segmentos,
garantindo perfeito alinhamento entre Features Dataset e Embeddings Dataset.
"""

import numpy as np
import torch
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import welch

from src.utils.sliding_windows import filter_valid_windows
from src.utils.embeddings_extractor import load_model, resample_to_30hz_5s
from src.utils.constants import (
    COL_ACC_X, COL_ACC_Y, COL_ACC_Z,
    COL_GYRO_X, COL_GYRO_Y, COL_GYRO_Z,
    COL_MAG_X, COL_MAG_Y, COL_MAG_Z
)


# ============================================================================
# FEATURES DO DOMÍNIO TEMPORAL
# ============================================================================

def mean_feature(signal):
    """Média do sinal."""
    return np.mean(signal)


def median_feature(signal):
    """Mediana do sinal."""
    return np.median(signal)


def std_feature(signal):
    """Desvio padrão do sinal."""
    return np.std(signal)


def variance_feature(signal):
    """Variância do sinal."""
    return np.var(signal)


def min_feature(signal):
    """Valor mínimo do sinal."""
    return np.min(signal)


def max_feature(signal):
    """Valor máximo do sinal."""
    return np.max(signal)


def range_feature(signal):
    """Range (max - min) do sinal."""
    return np.max(signal) - np.min(signal)


def mad_feature(signal):
    """Mean Absolute Deviation (MAD)."""
    return np.mean(np.abs(signal - np.mean(signal)))


def iqr_feature(signal):
    """Interquartile Range (IQR) - Q3 - Q1."""
    q75, q25 = np.percentile(signal, [75, 25])
    return q75 - q25


def skewness_feature(signal):
    """Assimetria (skewness) da distribuição."""
    return stats.skew(signal)


def kurtosis_feature(signal):
    """Curtose (kurtosis) da distribuição."""
    return stats.kurtosis(signal)


def rms_feature(signal):
    """Root Mean Square (RMS)."""
    return np.sqrt(np.mean(signal**2))


def zero_crossing_rate(signal):
    """Taxa de cruzamento por zero."""
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    return len(zero_crossings) / len(signal)


def mean_crossing_rate(signal):
    """
    Taxa de cruzamento pela média (Mean Crossing Rate).
    Similar ao ZCR, mas conta cruzamentos pela média do sinal.
    """
    mean_val = np.mean(signal)
    mean_crossings = np.where(np.diff(np.sign(signal - mean_val)))[0]
    return len(mean_crossings) / len(signal)


# ============================================================================
# FEATURES DO DOMÍNIO ESPECTRAL
# ============================================================================

def dominant_frequency(signal, sampling_rate=50):
    """
    Frequência dominante (com maior energia).
    
    Remove a componente DC (média) antes da FFT para evitar que 0 Hz
    seja sempre a frequência dominante.
    
    Args:
        signal: Sinal temporal
        sampling_rate: Taxa de amostragem em Hz
    
    Returns:
        float: Frequência dominante em Hz
    """
    # Remove componente DC (centrar o sinal em zero)
    signal_centered = signal - np.mean(signal)
    
    # FFT do sinal centrado
    n = len(signal_centered)
    yf = fft(signal_centered)
    xf = fftfreq(n, 1/sampling_rate)
    
    # Considera apenas frequências positivas (excluindo DC em 0 Hz)
    positive_freqs = xf[1:n//2]  # Começa em índice 1 para excluir 0 Hz
    power = np.abs(yf[1:n//2])**2
    
    # Frequência com maior potência
    if len(power) > 0 and np.max(power) > 0:
        dominant_idx = np.argmax(power)
        return positive_freqs[dominant_idx]
    return 0.0


def spectral_energy(signal, sampling_rate=50):
    """
    Energia espectral total.
    
    Returns:
        float: Soma da potência espectral
    """
    n = len(signal)
    yf = fft(signal)
    power = np.abs(yf[:n//2])**2
    return np.sum(power)


def spectral_entropy(signal, sampling_rate=50):
    """
    Entropia espectral (mede regularidade do espectro).
    
    Returns:
        float: Entropia espectral (0 = regular, alto = irregular)
    """
    # Calcula densidade espectral de potência
    freqs, psd = welch(signal, fs=sampling_rate, nperseg=min(256, len(signal)))
    
    # Normaliza para obter probabilidades
    psd_norm = psd / np.sum(psd)
    
    # Calcula entropia de Shannon
    # Remove zeros para evitar log(0)
    psd_norm = psd_norm[psd_norm > 0]
    entropy = -np.sum(psd_norm * np.log2(psd_norm))
    
    return entropy


def spectral_centroid(signal, sampling_rate=50):
    """Centro de massa do espectro."""
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1/sampling_rate)
    
    positive_freqs = xf[:n//2]
    power = np.abs(yf[:n//2])**2
    
    if np.sum(power) > 0:
        return np.sum(positive_freqs * power) / np.sum(power)
    return 0.0


# ============================================================================
# FEATURES ADICIONAIS DO ARTIGO (Movement Intensity, SMA, EVA, etc.)
# ============================================================================

def movement_intensity_features(acc_x, acc_y, acc_z):
    """
    Movement Intensity (MI), Average Intensity (AI) e Variance Intensity (VI).
    MI é o módulo euclidiano da aceleração (removendo gravidade estática).
    
    Args:
        acc_x, acc_y, acc_z: Sinais de aceleração nos 3 eixos
        
    Returns:
        dict: {'ai': Average Intensity, 'vi': Variance Intensity}
    """
    # Calcula MI para cada amostra (Euclidean norm)
    mi = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    # Average Intensity (AI)
    ai = np.mean(mi)
    
    # Variance Intensity (VI)
    vi = np.var(mi)
    
    return {'ai': ai, 'vi': vi}


def sma_feature(acc_x, acc_y, acc_z):
    """
    Normalized Signal Magnitude Area (SMA).
    Soma das magnitudes de aceleração nos 3 eixos, normalizada pelo tamanho da janela.
    
    Returns:
        float: SMA normalizado
    """
    T = len(acc_x)
    sma = (np.sum(np.abs(acc_x)) + np.sum(np.abs(acc_y)) + np.sum(np.abs(acc_z))) / T
    return sma


def eva_features(acc_x, acc_y, acc_z):
    """
    Eigenvalues of Dominant Directions (EVA).
    Calcula os eigenvalues da matriz de covariância das acelerações.
    Os eigenvalues capturam a direção e magnitude dos movimentos dominantes.
    
    Returns:
        dict: {'eva1': largest eigenvalue, 'eva2': 2nd, 'eva3': 3rd}
    """
    # Matriz de dados [n_samples, 3]
    acc_matrix = np.column_stack([acc_x, acc_y, acc_z])
    
    # Matriz de covariância
    cov_matrix = np.cov(acc_matrix.T)
    
    # Eigenvalues (já retornam em ordem decrescente)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Ordem decrescente
    
    return {
        'eva1': eigenvalues[0],
        'eva2': eigenvalues[1],
        'eva3': eigenvalues[2]
    }


def cagh_feature(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z):
    """
    Correlation between Acceleration along Gravity and Heading Directions (CAGH).
    Correlação entre aceleração na direção da gravidade e da heading direction.
    
    Simplificação: Assume que gravidade é predominantemente no eixo Z do acelerômetro,
    e heading é aproximado pelo gyro Z (yaw).
    
    Returns:
        float: Coeficiente de correlação
    """
    # Aproximação: aceleração total vs rotação em Z
    acc_total = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    if np.std(acc_total) > 0 and np.std(gyro_z) > 0:
        corr = np.corrcoef(acc_total, gyro_z)[0, 1]
        return corr if not np.isnan(corr) else 0.0
    return 0.0


def avh_feature(acc_y, acc_z, sampling_rate=50):
    """
    Averaged Velocity along Heading Direction (AVH).
    Velocidade média ao longo da direção de heading (aproximada por integração).
    
    Args:
        acc_y, acc_z: Acelerações Y e Z (plano horizontal de movimento)
        sampling_rate: Taxa de amostragem
    
    Returns:
        float: Velocidade média (magnitude)
    """
    dt = 1.0 / sampling_rate
    
    # Integra aceleração para obter velocidade
    vel_y = np.cumsum(acc_y) * dt
    vel_z = np.cumsum(acc_z) * dt
    
    # Magnitude da velocidade
    vel_magnitude = np.sqrt(vel_y**2 + vel_z**2)
    
    return np.mean(vel_magnitude)


def avg_feature(acc_x, acc_y, acc_z, sampling_rate=50):
    """
    Averaged Velocity along Gravity Direction (AVG).
    Velocidade média ao longo da direção da gravidade (eixo vertical).
    
    Returns:
        float: Velocidade média vertical
    """
    dt = 1.0 / sampling_rate
    
    # Integra aceleração vertical para obter velocidade
    vel_x = np.cumsum(acc_x) * dt
    vel_y = np.cumsum(acc_y) * dt
    vel_z = np.cumsum(acc_z) * dt
    
    # Magnitude da velocidade total
    vel_magnitude = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
    
    return np.mean(vel_magnitude)


def aratg_feature(gyro_x, gyro_y, gyro_z, sampling_rate=50):
    """
    Averaged Rotation Angles related to Gravity Direction (ARATG).
    Ângulos de rotação médios relacionados com a gravidade.
    
    Returns:
        float: Ângulo de rotação acumulado médio
    """
    dt = 1.0 / sampling_rate
    
    # Integra velocidades angulares para obter ângulos
    angle_x = np.cumsum(gyro_x) * dt
    angle_y = np.cumsum(gyro_y) * dt
    angle_z = np.cumsum(gyro_z) * dt
    
    # Magnitude dos ângulos
    angle_magnitude = np.sqrt(angle_x**2 + angle_y**2 + angle_z**2)
    
    return np.mean(angle_magnitude)


def aae_feature(acc_x, acc_y, acc_z):
    """
    Averaged Acceleration Energy (AAE).
    Energia média da aceleração sobre os 3 eixos.
    
    Returns:
        float: Energia média de aceleração
    """
    energy_x = np.mean(acc_x**2)
    energy_y = np.mean(acc_y**2)
    energy_z = np.mean(acc_z**2)
    
    return (energy_x + energy_y + energy_z) / 3.0


def are_feature(gyro_x, gyro_y, gyro_z):
    """
    Averaged Rotation Energy (ARE).
    Energia média de rotação sobre os 3 eixos do giroscópio.
    
    Returns:
        float: Energia média de rotação
    """
    energy_x = np.mean(gyro_x**2)
    energy_y = np.mean(gyro_y**2)
    energy_z = np.mean(gyro_z**2)
    
    return (energy_x + energy_y + energy_z) / 3.0


# ============================================================================
# EXTRAÇÃO COMPLETA DE FEATURES
# ============================================================================

def extract_temporal_features(signal):
    """
    Extrai todas as features temporais de um sinal (Tabela 1).
    
    Returns:
        dict: Dicionário com todas as features temporais
    """
    return {
        'mean': mean_feature(signal),
        'median': median_feature(signal),
        'std': std_feature(signal),
        'variance': variance_feature(signal),
        'min': min_feature(signal),
        'max': max_feature(signal),
        'range': range_feature(signal),
        'rms': rms_feature(signal),
        'mad': mad_feature(signal),
        'iqr': iqr_feature(signal),
        'skewness': skewness_feature(signal),
        'kurtosis': kurtosis_feature(signal),
        'zcr': zero_crossing_rate(signal),
        'mcr': mean_crossing_rate(signal)
    }


def extract_spectral_features(signal, sampling_rate=50):
    """
    Extrai todas as features espectrais de um sinal.
    
    Returns:
        dict: Dicionário com todas as features espectrais
    """
    return {
        'dominant_freq': dominant_frequency(signal, sampling_rate),
        'spectral_energy': spectral_energy(signal, sampling_rate),
        'spectral_entropy': spectral_entropy(signal, sampling_rate),
        'spectral_centroid': spectral_centroid(signal, sampling_rate)
    }


def extract_window_features(window_data, sampling_rate=50):
    """
    Extrai features completas de uma janela de dados.
    
    Para cada eixo individual dos 3 sensores (9 eixos no total), extrai:
    - 14 features temporais (Tabela 1)
    - 4 features espectrais
    
    Além disso, extrai 12 features multi-sensor:
    - AI, VI: Movement Intensity features
    - SMA: Signal Magnitude Area
    - EVA1, EVA2, EVA3: eigenvalues
    - CAGH, AVH, AVG, ARATG, AAE, ARE
    
    Total: 9 eixos × 18 features + 12 multi-sensor = 162 + 12 = 174 features
    
    EIXOS EXTRAÍDOS:
    - Acelerómetro: acc_x, acc_y, acc_z
    - Giroscópio: gyro_x, gyro_y, gyro_z
    - Magnetómetro: mag_x, mag_y, mag_z
    
    Args:
        window_data: Array [n_samples, 12] de uma janela
        sampling_rate: Taxa de amostragem em Hz
    
    Returns:
        dict: Features extraídas organizadas por eixo e tipo
    """
    features = {}
    
    # Extrai sinais dos eixos individuais usando constantes
    acc_x = window_data[:, COL_ACC_X]
    acc_y = window_data[:, COL_ACC_Y]
    acc_z = window_data[:, COL_ACC_Z]
    gyro_x = window_data[:, COL_GYRO_X]
    gyro_y = window_data[:, COL_GYRO_Y]
    gyro_z = window_data[:, COL_GYRO_Z]
    mag_x = window_data[:, COL_MAG_X]
    mag_y = window_data[:, COL_MAG_Y]
    mag_z = window_data[:, COL_MAG_Z]
    
    # ========================================================================
    # FEATURES POR EIXO (14 temporais + 4 espectrais = 18 por eixo)
    # ========================================================================
    sensor_axes = {
        'acc_x': acc_x, 'acc_y': acc_y, 'acc_z': acc_z,
        'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z,
        'mag_x': mag_x, 'mag_y': mag_y, 'mag_z': mag_z
    }
    
    for axis_name, signal in sensor_axes.items():
        # Features temporais (14)
        temporal = extract_temporal_features(signal)
        for feat_name, feat_value in temporal.items():
            features[f"{axis_name}_{feat_name}"] = feat_value
        
        # Features espectrais (4)
        spectral = extract_spectral_features(signal, sampling_rate)
        for feat_name, feat_value in spectral.items():
            features[f"{axis_name}_{feat_name}"] = feat_value
    
    # ========================================================================
    # FEATURES MULTI-SENSOR (11 features adicionais)
    # ========================================================================
    
    # Movement Intensity (AI, VI)
    mi_feats = movement_intensity_features(acc_x, acc_y, acc_z)
    features['ai'] = mi_feats['ai']
    features['vi'] = mi_feats['vi']
    
    # SMA
    features['sma'] = sma_feature(acc_x, acc_y, acc_z)
    
    # EVA (3 eigenvalues)
    eva_feats = eva_features(acc_x, acc_y, acc_z)
    features['eva1'] = eva_feats['eva1']
    features['eva2'] = eva_feats['eva2']
    features['eva3'] = eva_feats['eva3']
    
    # CAGH
    features['cagh'] = cagh_feature(acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
    
    # AVH
    features['avh'] = avh_feature(acc_y, acc_z, sampling_rate)
    
    # AVG
    features['avg'] = avg_feature(acc_x, acc_y, acc_z, sampling_rate)
    
    # ARATG
    features['aratg'] = aratg_feature(gyro_x, gyro_y, gyro_z, sampling_rate)
    
    # AAE
    features['aae'] = aae_feature(acc_x, acc_y, acc_z)
    
    # ARE
    features['are'] = are_feature(gyro_x, gyro_y, gyro_z)
    
    return features


def extract_features_from_windows(windows, sampling_rate=50, 
                                   extract_embeddings=True, device="cpu", batch_size=32):
    """
    Extrai features handcrafted e embeddings de todas as janelas válidas.
    
    Args:
        windows: Lista de janelas (de create_sliding_windows)
        sampling_rate: Taxa de amostragem em Hz
        extract_embeddings: Se True, extrai embeddings usando harnet5
        device: Device para computação de embeddings ('cpu' ou 'cuda')
        batch_size: Tamanho do batch para extração de embeddings
    
    Returns:
        tuple: (feature_matrix, labels, metadata, feature_names, embeddings)
            - feature_matrix: Array [n_windows, 174] features handcrafted
            - labels: Array [n_windows] com IDs das atividades
            - metadata: Lista de dicts com info de cada janela
            - feature_names: Lista com nomes das features
            - embeddings: Array [n_windows, 512] ou None se extract_embeddings=False
    """
    # Filtra janelas válidas
    valid_windows = filter_valid_windows(windows)
    n_windows = len(valid_windows)
    
    # Carrega modelo harnet5 se necessário
    feature_encoder = None
    if extract_embeddings:
        feature_encoder = load_model()
        if device != "cpu":
            feature_encoder.to(device)
    
    # Lista para armazenar features e segmentos
    features_list = []
    labels = []
    metadata = []
    segments_acc = []  # Para embeddings
    
    # Extrai features handcrafted
    for i, window in enumerate(valid_windows):
        # Extrai features da janela
        features = extract_window_features(window['data'], sampling_rate)
        
        # Converte para array (ordem consistente)
        feature_names = sorted(features.keys())
        feature_vector = np.array([features[name] for name in feature_names])
        
        features_list.append(feature_vector)
        labels.append(window['activity'])
        metadata.append({
            'activity': window['activity'],
            'device': window['device'],
            'participant_id': window.get('participant_id', -1),
            'start_idx': window['start_idx'],
            'end_idx': window['end_idx']
        })
        
        # Guarda aceleração para embeddings
        if extract_embeddings:
            acc_xyz = window['data'][:, [COL_ACC_X, COL_ACC_Y, COL_ACC_Z]]
            segments_acc.append(acc_xyz)
    
    # Converte para arrays numpy
    feature_matrix = np.array(features_list)
    labels = np.array(labels)
    
    # Extrai embeddings se solicitado
    embeddings = None
    if extract_embeddings and feature_encoder is not None:
        embeddings = _extract_embeddings_batch(
            segments=segments_acc,
            feature_encoder=feature_encoder,
            fs_in_hz=sampling_rate,
            batch_size=batch_size,
            device=device
        )
    
    return feature_matrix, labels, metadata, feature_names, embeddings


def _extract_embeddings_batch(segments, feature_encoder, fs_in_hz, batch_size, device):
    """
    Extrai embeddings de uma lista de segmentos usando harnet5.
    
    Args:
        segments: Lista de arrays [n_samples, 3] com aceleração X, Y, Z
        feature_encoder: Modelo harnet5 carregado
        fs_in_hz: Frequência de amostragem original em Hz
        batch_size: Tamanho do batch para processamento
        device: Device para computação
        
    Returns:
        np.ndarray: Embeddings [n_segments, 512]
    """
    # Reamostra todos os segmentos para 30Hz (requerido pelo modelo)
    resampled_segments = [resample_to_30hz_5s(seg, fs_in_hz)[0] for seg in segments]
    
    # Converte para array [n_segments, time, channels]
    x_all = np.array(resampled_segments, dtype=np.float32)
    
    # Transforma para formato do modelo: [n_segments, channels, time]
    x_all = np.transpose(x_all, (0, 2, 1))
    
    # Extrai embeddings em batches (silenciosamente)
    embeddings_list = []
    
    with torch.no_grad():
        for i in range(0, len(x_all), batch_size):
            batch_data = x_all[i:i+batch_size]
            batch_tensor = torch.from_numpy(batch_data).float().to(device)
            batch_embeddings = feature_encoder(batch_tensor)
            embeddings_list.append(batch_embeddings.cpu().numpy())
    
    # Concatena todos os embeddings
    embeddings = np.concatenate(embeddings_list, axis=0)
    
    # Remove dimensões extras se existirem
    if embeddings.ndim > 2:
        embeddings = embeddings.squeeze()
    
    return embeddings


def get_feature_names():
    """
    Retorna lista de nomes de todas as features extraídas.
    
    Total: 9 eixos × 18 features (14 temp + 4 spec) + 12 multi-sensor = 174 features
    
    Returns:
        list: Nomes das features na ordem de extração
    """
    axis_names = ['acc_x', 'acc_y', 'acc_z', 
                  'gyro_x', 'gyro_y', 'gyro_z',
                  'mag_x', 'mag_y', 'mag_z']
    
    # 14 features temporais (Tabela 1)
    temporal_features = ['mean', 'median', 'std', 'variance', 'min', 'max', 
                        'range', 'rms', 'mad', 'iqr', 'skewness', 'kurtosis', 'zcr', 'mcr']
    
    # 4 features espectrais
    spectral_features = ['dominant_freq', 'spectral_energy', 'spectral_entropy', 
                        'spectral_centroid']
    
    feature_names = []
    
    # Features por eixo (9 × 18 = 162)
    for axis in axis_names:
        for temp_feat in temporal_features:
            feature_names.append(f"{axis}_{temp_feat}")
        for spec_feat in spectral_features:
            feature_names.append(f"{axis}_{spec_feat}")
    
    # Features multi-sensor (12)
    multi_sensor_features = ['ai', 'vi', 'sma', 'eva1', 'eva2', 'eva3', 
                            'cagh', 'avh', 'avg', 'aratg', 'aae', 'are']
    feature_names.extend(multi_sensor_features)
    
    return feature_names

