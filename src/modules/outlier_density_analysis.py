"""
Análise de densidade de outliers usando o método IQR (Interquartile Range).
Calcula e visualiza a percentagem de outliers por atividade e sensor.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sem interface gráfica
import matplotlib.pyplot as plt
import os

from src.utils.sensor_calculations import calculate_sensor_modules


def detect_outliers_iqr(data):
    """
    Deteta outliers usando o método IQR (definição de Tukey).
    Outliers: valores fora do intervalo [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    """
    # Calcula limites IQR
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    # Define limites para outliers (Tukey)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identifica outliers (valores fora dos limites)
    outliers = (data < lower_bound) | (data > upper_bound)
    
    return outliers

def calculate_outlier_density(data, participant_id="todos_participantes", output_dir="plots"):
    """
    Calcula a densidade de outliers para cada atividade usando apenas sensores do pulso direito.
    
    Args:
        data (numpy.ndarray): Dados carregados (pode ser de um participante ou todos)
        participant_id (str/int): ID do participante ou "todos_participantes" (para salvamento)
        output_dir (str): Diretório onde salvar o gráfico
    
    Returns:
        dict: Dicionário com densidades de outliers por atividade e sensor
    """
    # Filtra dados apenas do pulso direito (device_id = 2)
    right_wrist_data = data[data[:, 0] == 2]
    
    if len(right_wrist_data) == 0:
        print("Aviso: Nenhum dado encontrado para o pulso direito")
        return {}
    
    # Calcula os módulos dos sensores
    modules = calculate_sensor_modules(right_wrist_data)
    activities = right_wrist_data[:, 11]  # Coluna 12: Activity Label
    
    # Mapeamento de atividades
    activity_names = {
        1: "Stand", 2: "Sit", 3: "Sit and Talk", 4: "Walk", 5: "Walk and Talk",
        6: "Climb Stair", 7: "Climb Stair and Talk", 8: "Stand->Sit", 9: "Sit->Stand",
        10: "Stand->Sit and Talk", 11: "Sit->Stand and Talk", 12: "Stand->Walk",
        13: "Walk->Stand", 14: "Stand->Climb Stair", 15: "Climb Stair->Walk",
        16: "Climb Stair and Talk->Walk and Talk"
    }
    
    # Dicionário para armazenar resultados
    results = {
        'activities': [],
        'acc_densities': [],
        'gyro_densities': [],
        'mag_densities': [],
        'activity_names': []
    }
    
    # Para cada atividade
    for activity_id in sorted(np.unique(activities)):
        activity_mask = activities == activity_id
        activity_data = {
            'acc': modules['acc_module'][activity_mask],
            'gyro': modules['gyro_module'][activity_mask],
            'mag': modules['mag_module'][activity_mask]
        }
        
        # Calcula densidade de outliers para cada sensor
        densities = {}
        for sensor_name, sensor_data in activity_data.items():
            if len(sensor_data) > 0:
                outliers = detect_outliers_iqr(sensor_data)
                n_outliers = np.sum(outliers)
                n_total = len(sensor_data)
                density = (n_outliers / n_total) * 100 if n_total > 0 else 0
                densities[sensor_name] = density
            else:
                densities[sensor_name] = 0
        
        # Armazena resultados
        results['activities'].append(activity_id)
        results['acc_densities'].append(densities['acc'])
        results['gyro_densities'].append(densities['gyro'])
        results['mag_densities'].append(densities['mag'])
        results['activity_names'].append(activity_names.get(activity_id, f"Atividade {activity_id}"))
    
    # Estatísticas resumidas
    print(f"Densidade média de outliers: Acc={np.mean(results['acc_densities']):.1f}% Gyro={np.mean(results['gyro_densities']):.1f}% Mag={np.mean(results['mag_densities']):.1f}%")
    
    # Cria visualização
    create_outlier_density_plot(results, participant_id, output_dir)
    
    return results

def create_outlier_density_plot(results, participant_id, output_dir="plots"):
    """
    Cria gráfico de barras com as densidades de outliers por atividade.
    
    Args:
        results (dict): Resultados da análise de densidade
        participant_id (str/int): ID do participante ou "todos_participantes"
    """
    # Título adaptativo
    if participant_id == "todos_participantes":
        title = 'Densidade de Outliers por Atividade - Pulso Direito (Todos os Participantes)'
    else:
        title = f'Densidade de Outliers por Atividade - Pulso Direito (Participante {participant_id})'
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    sensors = ['acc', 'gyro', 'mag']
    sensor_names = ['Acelerómetro', 'Giroscópio', 'Magnetómetro']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (sensor, sensor_name, color) in enumerate(zip(sensors, sensor_names, colors)):
        ax = axes[i]
        
        # Dados para o gráfico
        densities = results[f'{sensor}_densities']
        activities = results['activity_names']
        
        # Cria gráfico de barras
        bars = ax.bar(range(len(activities)), densities, color=color, alpha=0.7)
        
        # Configura o gráfico
        ax.set_title(f'{sensor_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Atividade', fontsize=12)
        ax.set_ylabel('Densidade de Outliers (%)', fontsize=12)
        ax.set_xticks(range(len(activities)))
        ax.set_xticklabels([f"{i+1}" for i in range(len(activities))], rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Adiciona valores nas barras
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Salva o gráfico
    os.makedirs(output_dir, exist_ok=True)
    
    if participant_id == "todos_participantes":
        filename = 'outlier_density_todos_participantes.png'
    else:
        filename = f'outlier_density_participant_{participant_id}.png'
    
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()  # Fecha a figura para liberar memória
    
    print(f"\nGráfico salvo em: {filepath}")

def analyze_outlier_patterns(results):
    """
    Analisa padrões nos outliers e fornece insights.
    
    Args:
        results (dict): Resultados da análise de densidade
    """
    # Calcula médias por tipo de atividade
    static_activities = [1, 2, 3]
    dynamic_activities = [4, 5, 6, 7]
    transition_activities = [8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    def calc_group_avg(activity_ids):
        group_indices = [i for i, act_id in enumerate(results['activities']) if act_id in activity_ids]
        if group_indices:
            return {
                'acc': np.mean([results['acc_densities'][i] for i in group_indices]),
                'gyro': np.mean([results['gyro_densities'][i] for i in group_indices]),
                'mag': np.mean([results['mag_densities'][i] for i in group_indices])
            }
        return None
    
    static_avg = calc_group_avg(static_activities)
    dynamic_avg = calc_group_avg(dynamic_activities)
    transition_avg = calc_group_avg(transition_activities)
    
    # Print resumo minimalista
    print("\nMédias por tipo de atividade:")
    if static_avg:
        print(f"  Estáticas:   Acc={static_avg['acc']:.1f}%  Gyro={static_avg['gyro']:.1f}%  Mag={static_avg['mag']:.1f}%")
    if dynamic_avg:
        print(f"  Dinâmicas:   Acc={dynamic_avg['acc']:.1f}%  Gyro={dynamic_avg['gyro']:.1f}%  Mag={dynamic_avg['mag']:.1f}%")
    if transition_avg:
        print(f"  Transições:  Acc={transition_avg['acc']:.1f}%  Gyro={transition_avg['gyro']:.1f}%  Mag={transition_avg['mag']:.1f}%")
