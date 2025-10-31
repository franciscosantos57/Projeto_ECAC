"""
Funções auxiliares para segmentação de dados com sliding windows.
Implementa janelas deslizantes com overlap para extração de features.
"""

import numpy as np


def create_sliding_windows(data, window_size_sec=5, overlap=0.5, sampling_rate=50):
    """
    Cria janelas deslizantes (sliding windows) sobre os dados.
    
    Args:
        data: Array com dados de sensores [n_samples, n_features]
        window_size_sec: Tamanho da janela em segundos (padrão: 5s)
        overlap: Overlap entre janelas (0.5 = 50%)
        sampling_rate: Taxa de amostragem em Hz (padrão: 50 Hz)
    
    Returns:
        list: Lista de dicionários com informação de cada janela:
              {'data': array, 'start_idx': int, 'end_idx': int, 
               'activity': int, 'device': int, 'is_valid': bool}
    """
    # Calcula tamanho da janela em amostras
    window_size = int(window_size_sec * sampling_rate)
    step_size = int(window_size * (1 - overlap))
    
    windows = []
    n_samples = len(data)
    
    # Itera sobre os dados com sliding window
    start_idx = 0
    while start_idx + window_size <= n_samples:
        end_idx = start_idx + window_size
        window_data = data[start_idx:end_idx]
        
        # Verifica se a janela cobre apenas uma atividade
        activities_in_window = np.unique(window_data[:, 11])
        is_single_activity = len(activities_in_window) == 1
        
        # Verifica se a janela usa apenas um dispositivo
        devices_in_window = np.unique(window_data[:, 0])
        is_single_device = len(devices_in_window) == 1
        
        # Janela válida: uma única atividade e um único dispositivo
        is_valid = is_single_activity and is_single_device
        
        window_info = {
            'data': window_data,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'activity': int(activities_in_window[0]) if is_single_activity else -1,
            'device': int(devices_in_window[0]) if is_single_device else -1,
            'is_valid': is_valid
        }
        
        windows.append(window_info)
        start_idx += step_size
    
    return windows


def filter_valid_windows(windows):
    """
    Filtra apenas janelas válidas (uma atividade, um dispositivo).
    
    Args:
        windows: Lista de janelas retornada por create_sliding_windows
    
    Returns:
        list: Lista apenas com janelas válidas
    """
    return [w for w in windows if w['is_valid']]


def get_window_statistics(windows):
    """
    Calcula estatísticas sobre as janelas criadas.
    
    Returns:
        dict: Estatísticas (total, válidas, descartadas, por atividade, etc.)
    """
    total_windows = len(windows)
    valid_windows = [w for w in windows if w['is_valid']]
    n_valid = len(valid_windows)
    n_discarded = total_windows - n_valid
    
    # Conta por atividade
    activity_counts = {}
    for w in valid_windows:
        activity = w['activity']
        activity_counts[activity] = activity_counts.get(activity, 0) + 1
    
    # Conta por dispositivo
    device_counts = {}
    for w in valid_windows:
        device = w['device']
        device_counts[device] = device_counts.get(device, 0) + 1
    
    return {
        'total_windows': total_windows,
        'valid_windows': n_valid,
        'discarded_windows': n_discarded,
        'discard_rate': (n_discarded / total_windows * 100) if total_windows > 0 else 0,
        'windows_per_activity': activity_counts,
        'windows_per_device': device_counts
    }
