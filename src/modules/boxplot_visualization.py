"""
Visualização de boxplots dos módulos dos sensores.
Separados por atividade e dispositivo.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
matplotlib.use('Agg')

from src.utils.sensor_calculations import calculate_sensor_modules


def create_boxplot_visualization(data, participant_id="todos_participantes", output_dir="plots"):
    """
    Cria boxplots para os módulos dos sensores separados por atividade e dispositivo.
    
    Args:
        data: Dados carregados
        participant_id: ID do participante ou "todos_participantes"
        output_dir: Diretório onde guardar o gráfico
    """
    
    # Calcula módulos dos sensores
    modules = calculate_sensor_modules(data)
    
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
    
    # Título adaptativo
    if participant_id == "todos_participantes":
        title = 'Boxplots dos Módulos dos Sensores - Todos os Participantes Combinados'
    else:
        title = f'Boxplots dos Módulos dos Sensores - Participante {participant_id}'
    
    # Cria figura com subplots
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Cores por dispositivo
    device_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Tipos de sensor (3 linhas)
    sensor_types = ['acc_module', 'gyro_module', 'mag_module']
    sensor_titles = ['Módulo do Acelerómetro', 'Módulo do Giroscópio', 'Módulo do Magnetómetro']
    
    for sensor_idx, (sensor_type, sensor_title) in enumerate(zip(sensor_types, sensor_titles)):
        for device_id in range(1, 6):
            ax = axes[sensor_idx, device_id - 1]
            
            # Filtra dados do dispositivo
            device_data = data[data[:, 0] == device_id]
            device_modules = modules[sensor_type][data[:, 0] == device_id]
            device_activities = device_data[:, 11]
            
            # Agrupa por atividade
            boxplot_data = []
            activity_labels = []
            
            for activity_id in sorted(np.unique(device_activities)):
                activity_mask = device_activities == activity_id
                activity_modules = device_modules[activity_mask]
                
                if len(activity_modules) > 0:
                    boxplot_data.append(activity_modules)
                    activity_labels.append(f"{int(activity_id)}")
            
            # Cria boxplot
            if boxplot_data:
                bp = ax.boxplot(boxplot_data, patch_artist=True, labels=activity_labels)
                
                # Aplica cores
                for patch in bp['boxes']:
                    patch.set_facecolor(device_colors[device_id - 1])
                    patch.set_alpha(0.7)
                
                # Estiliza elementos
                for element in ['whiskers', 'fliers', 'medians', 'caps']:
                    plt.setp(bp[element], color='black', linewidth=1)
                
                # Configura eixo
                ax.set_title(f'{device_names[device_id]}', fontsize=10, fontweight='bold')
                ax.set_xlabel('Atividade', fontsize=8)
                ax.set_ylabel('Módulo', fontsize=8)
                ax.tick_params(axis='x', rotation=45, labelsize=7)
                ax.tick_params(axis='y', labelsize=7)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Sem dados', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{device_names[device_id]}', fontsize=10, fontweight='bold')
        
        # Título da linha
        axes[sensor_idx, 0].text(-0.2, 0.5, sensor_title, rotation=90, 
                                ha='center', va='center', transform=axes[sensor_idx, 0].transAxes,
                                fontsize=12, fontweight='bold')
    
    # Ajusta layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.93, bottom=0.15)
    
    # Guarda figura
    os.makedirs(output_dir, exist_ok=True)
    
    if participant_id == "todos_participantes":
        filename = 'boxplot_todos_participantes.png'
    else:
        filename = f'boxplot_participant_{participant_id}.png'
    
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Boxplot guardado em: {filepath}")
    return fig

