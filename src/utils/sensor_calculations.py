"""
Funções auxiliares para cálculos relacionados com sensores.
Inclui cálculo de módulos e normalização de dados.
"""

import numpy as np
from src.utils.constants import (
    COL_ACC_X, COL_ACC_Y, COL_ACC_Z,
    COL_GYRO_X, COL_GYRO_Y, COL_GYRO_Z,
    COL_MAG_X, COL_MAG_Y, COL_MAG_Z
)


def calculate_sensor_modules(data):
    """
    Calcula os módulos dos vetores de aceleração, giroscópio e magnetómetro.
    
    Args:
        data (numpy.ndarray): Dados carregados com formato [n_samples, n_features]
                             Colunas: [device_id, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z,
                                      mag_x, mag_y, mag_z, timestamp, activity_label]
    
    Returns:
        dict: Dicionário com os módulos calculados:
              {'acc_module': array, 'gyro_module': array, 'mag_module': array}
    """
    # Extrai componentes dos sensores usando constantes
    acc_x, acc_y, acc_z = data[:, COL_ACC_X], data[:, COL_ACC_Y], data[:, COL_ACC_Z]
    gyro_x, gyro_y, gyro_z = data[:, COL_GYRO_X], data[:, COL_GYRO_Y], data[:, COL_GYRO_Z]
    mag_x, mag_y, mag_z = data[:, COL_MAG_X], data[:, COL_MAG_Y], data[:, COL_MAG_Z]
    
    # Calcula módulos: ||v|| = sqrt(x² + y² + z²)
    acc_module = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    gyro_module = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
    mag_module = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
    
    return {
        'acc_module': acc_module,
        'gyro_module': gyro_module,
        'mag_module': mag_module
    }


def zscore_normalization(data):
    """
    Normaliza os dados usando Z-Score.
    
    Fórmula: Z = (x - μ) / σ
    onde μ é a média e σ é o desvio padrão
    
    Args:
        data (numpy.ndarray): Dados a normalizar [n_samples, n_features]
    
    Returns:
        numpy.ndarray: Dados normalizados com mesma dimensão
    """
    # Calcula média e desvio padrão por coluna
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Evita divisão por zero
    std[std == 0] = 1.0
    
    # Aplica normalização Z-Score
    return (data - mean) / std
