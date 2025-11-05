"""
EXERCÍCIO 4.1: Análise de Significância Estatística
Determina a significância estatística dos valores médios dos módulos dos sensores
nas diferentes atividades usando testes apropriados (paramétricos ou não-paramétricos).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kstest, kruskal, f_oneway
import os

from src.utils.sensor_calculations import calculate_sensor_modules
from src.utils.constants import ACTIVITY_NAMES, DEVICE_NAMES, COL_ACTIVITY


def test_normality(data, sensor_name, activity_id):
    """
    Testa normalidade da distribuição usando Kolmogorov-Smirnov.
    
    Args:
        data: Array com dados do sensor para uma atividade
        sensor_name: Nome do sensor
        activity_id: ID da atividade
    
    Returns:
        tuple: (statistic, p_value, is_normal)
    """
    # Teste Kolmogorov-Smirnov com distribuição normal
    # H0: os dados seguem uma distribuição normal
    statistic, p_value = kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    
    # Critério: p > 0.05 indica normalidade
    is_normal = p_value > 0.05
    
    return statistic, p_value, is_normal


def analyze_statistical_significance(data, output_dir=None):
    """
    Analisa significância estatística dos valores médios entre atividades.
    
    Para cada sensor:
    1. Testa normalidade usando Kolmogorov-Smirnov
    2. Escolhe teste apropriado (ANOVA se normal, Kruskal-Wallis se não)
    3. Analisa diferenças significativas entre atividades
    
    Args:
        data: Dados completos de sensores
        output_dir: Diretório para salvar resultados (opcional, não usado no exercício 4)
    """
    
    # Calcula módulos dos sensores
    modules = calculate_sensor_modules(data)
    activities = data[:, COL_ACTIVITY]
    
    sensor_types = ['acc_module', 'gyro_module', 'mag_module']
    sensor_names = ['Acelerómetro', 'Giroscópio', 'Magnetómetro']
    
    results = {}
    
    for sensor_type, sensor_name in zip(sensor_types, sensor_names):

        
        # Agrupa dados por atividade
        activity_data = {}
        for activity_id in sorted(np.unique(activities)):
            activity_mask = activities == activity_id
            activity_data[activity_id] = modules[sensor_type][activity_mask]
        
        # 1. TESTE DE NORMALIDADE (Kolmogorov-Smirnov)
        normality_results = {}
        all_normal = True
        
        for activity_id, data_values in activity_data.items():
            if len(data_values) > 0:
                mean_val = np.mean(data_values)
                std_val = np.std(data_values)
                ks_stat, p_val, is_normal = test_normality(data_values, sensor_name, activity_id)
                
                normality_results[activity_id] = {
                    'n': len(data_values),
                    'mean': mean_val,
                    'std': std_val,
                    'ks_statistic': ks_stat,
                    'p_value': p_val,
                    'is_normal': is_normal
                }
                
                if not is_normal:
                    all_normal = False
        
        # 2. ESCOLHA DO TESTE ESTATÍSTICO
        if all_normal:
            test_type = "ANOVA"
        else:
            test_type = "Kruskal-Wallis"
        
        print(f"\n{sensor_name}: {test_type}", end="")
        
        # 3. TESTE DE SIGNIFICÂNCIA
        data_groups = [data_values for data_values in activity_data.values() if len(data_values) > 0]
        
        if test_type == "ANOVA":
            statistic, p_value = f_oneway(*data_groups)
        else:
            statistic, p_value = kruskal(*data_groups)
        
        is_significant = p_value < 0.05
        
        # Formatação do p-value: se for muito pequeno (< 1e-100), mostrar "< 1e-100"
        if p_value < 1e-100:
            p_str = "< 1e-100"
        elif p_value < 0.001:
            p_str = f"{p_value:.2e}"
        else:
            p_str = f"{p_value:.4f}"
        
        print(f" → p={p_str} ({'Significativo' if is_significant else 'Não significativo'})")
        
        # Armazena resultados
        results[sensor_type] = {
            'sensor_name': sensor_name,
            'normality': normality_results,
            'all_normal': all_normal,
            'test_type': test_type,
            'test_statistic': statistic,
            'p_value': p_value,
            'is_significant': is_significant
        }
    
    return results


