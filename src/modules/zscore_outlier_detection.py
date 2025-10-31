"""
Deteção de outliers usando o método Z-Score.
Implementa e compara diferentes thresholds (k=3, 3.5, 4) e compara com método IQR.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sem interface gráfica
import matplotlib.pyplot as plt
import os

from src.utils.sensor_calculations import calculate_sensor_modules


def detect_outliers_zscore(data, k=3):
    """
    EXERCÍCIO 3.3: Deteta outliers usando Z-Score.
    
    Z-Score mede desvios-padrão da média: Z = (x - μ) / σ
    Outliers: |Z| > k (tipicamente k=3)
    """
    # Calcula média e desvio padrão
    mean = np.mean(data)
    std = np.std(data)
    
    # Evita divisão por zero
    if std == 0:
        return np.zeros(len(data), dtype=bool)
    
    # Calcula Z-Score para cada amostra
    z_scores = np.abs((data - mean) / std)
    
    # Identifica outliers: |Z| > k
    outliers = z_scores > k
    
    return outliers

def create_zscore_plots(data, k_values=[3, 3.5, 4], output_dir="plots"):
    """
    EXERCÍCIO 3.4: Cria plots separados por atividade mostrando outliers detectados
    pelo Z-Score em vermelho e os restantes pontos em azul.
    
    IMPORTANTE: Os outliers são detetados POR ATIVIDADE, ou seja, o Z-Score é calculado
    separadamente para cada atividade. Isto significa que cada atividade tem o seu próprio
    limite de deteção, baseado na média e desvio padrão específicos dessa atividade.
    
    Args:
        data (numpy.ndarray): Dados combinados de todos os participantes
        k_values (list): Lista de valores de k para testar
        output_dir (str): Diretório base onde salvar os gráficos
    """
    # Calcula os módulos dos sensores
    modules = calculate_sensor_modules(data)
    activities = data[:, 11]  # Coluna 12: Activity Label
    devices = data[:, 0]  # Coluna 1: Device ID
    
    # Mapeamento de atividades
    activity_names = {
        1: "Stand", 2: "Sit", 3: "Sit and Talk", 4: "Walk", 5: "Walk and Talk",
        6: "Climb Stair", 7: "Climb Stair and Talk", 8: "Stand->Sit", 9: "Sit->Stand",
        10: "Stand->Sit and Talk", 11: "Sit->Stand and Talk", 12: "Stand->Walk",
        13: "Walk->Stand", 14: "Stand->Climb Stair", 15: "Climb Stair->Walk",
        16: "Climb Stair and Talk->Walk and Talk"
    }
    
    # Mapeamento de dispositivos
    device_names = {
        1: "Pulso Esquerdo", 2: "Pulso Direito", 3: "Peito", 
        4: "Perna Superior Direita", 5: "Perna Inferior Esquerda"
    }
    
    sensor_types = ['acc_module', 'gyro_module', 'mag_module']
    sensor_titles = ['Acelerómetro', 'Giroscópio', 'Magnetómetro']
    
    # Para cada valor de k
    for k in k_values:
        print(f"\nProcessando gráficos para k={k}...")
        
        # Cria uma figura grande com subplots
        fig, axes = plt.subplots(3, 5, figsize=(25, 15))
        fig.suptitle(f'Deteção de Outliers usando Z-Score (k={k}) - Todos os Participantes', 
                     fontsize=18, fontweight='bold')
        
        # Para cada tipo de sensor (3 linhas)
        for sensor_idx, (sensor_type, sensor_title) in enumerate(zip(sensor_types, sensor_titles)):
            
            # Para cada dispositivo (5 colunas)
            for device_id in range(1, 6):
                ax = axes[sensor_idx, device_id - 1]
                
                # Filtra dados para o dispositivo específico
                device_mask = devices == device_id
                device_modules = modules[sensor_type][device_mask]
                device_activities = activities[device_mask]
                
                # Para cada atividade, plota os pontos
                unique_activities = sorted(np.unique(device_activities))
                
                for activity_id in unique_activities:
                    activity_mask = device_activities == activity_id
                    
                    # Valores do módulo para esta atividade
                    activity_values = device_modules[activity_mask]
                    
                    # Deteta outliers usando Z-Score PARA ESTA ATIVIDADE ESPECÍFICA
                    outliers_in_activity = detect_outliers_zscore(activity_values, k=k)
                    
                    # Posição x (número da atividade)
                    x_position = activity_id
                    
                    # Plota não-outliers em azul
                    non_outliers = activity_values[~outliers_in_activity]
                    if len(non_outliers) > 0:
                        # Adiciona jitter para melhor visualização
                        x_jitter = np.random.normal(x_position, 0.1, len(non_outliers))
                        ax.scatter(x_jitter, non_outliers, c='blue', alpha=0.3, s=1, label='Normal' if activity_id == unique_activities[0] else '')
                    
                    # Plota outliers em vermelho
                    outliers = activity_values[outliers_in_activity]
                    if len(outliers) > 0:
                        x_jitter = np.random.normal(x_position, 0.1, len(outliers))
                        ax.scatter(x_jitter, outliers, c='red', alpha=0.6, s=3, label='Outlier' if activity_id == unique_activities[0] else '')
                
                # Configura o eixo
                ax.set_title(f'{device_names[device_id]}', fontsize=10, fontweight='bold')
                ax.set_xlabel('Atividade', fontsize=8)
                ax.set_ylabel('Módulo', fontsize=8)
                ax.set_xticks(unique_activities)
                ax.set_xticklabels([str(int(act)) for act in unique_activities], fontsize=6)
                ax.grid(True, alpha=0.3)
                
                # Adiciona legenda apenas no primeiro subplot
                if sensor_idx == 0 and device_id == 1:
                    ax.legend(loc='upper right', fontsize=8)
            
            # Adiciona título da linha (tipo de sensor)
            axes[sensor_idx, 0].text(-0.15, 0.5, sensor_title, rotation=90, 
                                    ha='center', va='center', transform=axes[sensor_idx, 0].transAxes,
                                    fontsize=14, fontweight='bold')
        
        # Ajusta o layout
        plt.tight_layout()
        plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.05)
        
        # Salva o gráfico em subpasta específica para k
        k_dir = os.path.join(output_dir, f'zscore_outliers_k{k}')
        os.makedirs(k_dir, exist_ok=True)
        filename = f'zscore_outliers_k{k}.png'
        filepath = os.path.join(k_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Gráfico salvo em: {filepath}")

def compare_methods(data, output_dir="plots"):
    """
    EXERCÍCIO 3.5: Compara os métodos IQR e Z-Score para os sensores do pulso direito.
    """
    print("Comparando métodos IQR vs Z-Score (pulso direito)\n")
    
    # Filtra dados apenas do pulso direito (device_id = 2)
    right_wrist_data = data[data[:, 0] == 2]
    
    if len(right_wrist_data) == 0:
        print("Aviso: Nenhum dado encontrado para o pulso direito")
        return
    
    # Calcula os módulos dos sensores
    modules = calculate_sensor_modules(right_wrist_data)
    activities = right_wrist_data[:, 11]
    
    # Mapeamento de atividades
    activity_names = {
        1: "Stand", 2: "Sit", 3: "Sit and Talk", 4: "Walk", 5: "Walk and Talk",
        6: "Climb Stair", 7: "Climb Stair and Talk", 8: "Stand->Sit", 9: "Sit->Stand",
        10: "Stand->Sit and Talk", 11: "Sit->Stand and Talk", 12: "Stand->Walk",
        13: "Walk->Stand", 14: "Stand->Climb Stair", 15: "Climb Stair->Walk",
        16: "Climb Stair and Talk->Walk and Talk"
    }
    
    sensors = ['acc_module', 'gyro_module', 'mag_module']
    sensor_names = ['Acelerómetro', 'Giroscópio', 'Magnetómetro']
    k_values = [3, 3.5, 4]
    
    # Tabela de comparação
    comparison_data = {}
    
    for sensor, sensor_name in zip(sensors, sensor_names):
        comparison_data[sensor] = {
            'activities': [],
            'iqr': [],
            'z3': [],
            'z35': [],
            'z4': []
        }
        
        for activity_id in sorted(np.unique(activities)):
            activity_mask = activities == activity_id
            activity_data = modules[sensor][activity_mask]
            
            if len(activity_data) > 0:
                # Método IQR
                from src.modules.outlier_density_analysis import detect_outliers_iqr
                outliers_iqr = detect_outliers_iqr(activity_data)
                density_iqr = (np.sum(outliers_iqr) / len(activity_data)) * 100
                
                # Método Z-Score com diferentes k
                outliers_z3 = detect_outliers_zscore(activity_data, k=3)
                density_z3 = (np.sum(outliers_z3) / len(activity_data)) * 100
                
                outliers_z35 = detect_outliers_zscore(activity_data, k=3.5)
                density_z35 = (np.sum(outliers_z35) / len(activity_data)) * 100
                
                outliers_z4 = detect_outliers_zscore(activity_data, k=4)
                density_z4 = (np.sum(outliers_z4) / len(activity_data)) * 100
                
                # Armazena dados
                comparison_data[sensor]['activities'].append(activity_names.get(activity_id, f"Atividade {activity_id}"))
                comparison_data[sensor]['iqr'].append(density_iqr)
                comparison_data[sensor]['z3'].append(density_z3)
                comparison_data[sensor]['z35'].append(density_z35)
                comparison_data[sensor]['z4'].append(density_z4)
        
        # Médias por sensor
        avg_iqr = np.mean(comparison_data[sensor]['iqr'])
        avg_z3 = np.mean(comparison_data[sensor]['z3'])
        avg_z35 = np.mean(comparison_data[sensor]['z35'])
        avg_z4 = np.mean(comparison_data[sensor]['z4'])
        print(f"{sensor_name}: IQR={avg_iqr:.1f}%  Z(k=3)={avg_z3:.1f}%  Z(k=3.5)={avg_z35:.1f}%  Z(k=4)={avg_z4:.1f}%")
    
    # Cria gráfico de comparação
    create_comparison_plot(comparison_data, sensor_names, output_dir)

def create_comparison_plot(comparison_data, sensor_names, output_dir="plots"):
    """
    Cria gráfico comparativo entre os métodos IQR e Z-Score.
    
    Args:
        comparison_data: Dados da comparação
        sensor_names: Nomes dos sensores
        output_dir: Diretório onde salvar o gráfico
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Comparação de Densidades de Outliers: IQR vs Z-Score - Pulso Direito', 
                 fontsize=16, fontweight='bold')
    
    sensors = ['acc_module', 'gyro_module', 'mag_module']
    
    for idx, (sensor, sensor_name) in enumerate(zip(sensors, sensor_names)):
        ax = axes[idx]
        
        activities = comparison_data[sensor]['activities']
        x = np.arange(len(activities))
        width = 0.18
        
        # Plota barras para cada método
        ax.bar(x - 1.5*width, comparison_data[sensor]['iqr'], width, label='IQR', alpha=0.8, color='#1f77b4')
        ax.bar(x - 0.5*width, comparison_data[sensor]['z3'], width, label='Z-Score (k=3)', alpha=0.8, color='#ff7f0e')
        ax.bar(x + 0.5*width, comparison_data[sensor]['z35'], width, label='Z-Score (k=3.5)', alpha=0.8, color='#2ca02c')
        ax.bar(x + 1.5*width, comparison_data[sensor]['z4'], width, label='Z-Score (k=4)', alpha=0.8, color='#d62728')
        
        ax.set_title(sensor_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Atividade', fontsize=11)
        ax.set_ylabel('Densidade de Outliers (%)', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i+1) for i in range(len(activities))], rotation=45, fontsize=8)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Salva o gráfico
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'comparison_iqr_vs_zscore.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nGráfico de comparação salvo em: {filepath}")
