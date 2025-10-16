import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
matplotlib.use('Agg')  # Backend sem interface gráfica

def calculate_sensor_modules(data):
    """
    Calcula os módulos dos vetores de aceleração, giroscópio e magnetómetro.
    
    Args:
        data (numpy.ndarray): Dados carregados com load_participant_data
    
    Returns:
        dict: Dicionário com os módulos calculados
              {'acc_module': array, 'gyro_module': array, 'mag_module': array}
    """
    # Índices das colunas (baseado na descrição do dataset)
    # Coluna 1: Device ID, Colunas 2-4: Accelerometer, Colunas 5-7: Gyroscope, 
    # Colunas 8-10: Magnetometer, Coluna 11: Timestamp, Coluna 12: Activity Label
    
    acc_x = data[:, 1]  # Coluna 2
    acc_y = data[:, 2]  # Coluna 3  
    acc_z = data[:, 3]  # Coluna 4
    
    gyro_x = data[:, 4]  # Coluna 5
    gyro_y = data[:, 5]  # Coluna 6
    gyro_z = data[:, 6]  # Coluna 7
    
    mag_x = data[:, 7]   # Coluna 8
    mag_y = data[:, 8]   # Coluna 9
    mag_z = data[:, 9]   # Coluna 10
    
    # Calcula os módulos: ||T|| = sqrt(tx² + ty² + tz²)
    acc_module = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    gyro_module = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
    mag_module = np.sqrt(mag_x**2 + mag_y**2 + mag_z**2)
    
    return {
        'acc_module': acc_module,
        'gyro_module': gyro_module,
        'mag_module': mag_module
    }

def create_boxplot_visualization(data, participant_id="todos_participantes"):
    """
    Cria boxplots para os módulos dos sensores separados por atividade e dispositivo.
    
    Args:
        data (numpy.ndarray): Dados carregados (pode ser de um participante ou todos)
        participant_id (str/int): ID do participante ou "todos_participantes" (para título)
    """
    
    # Calcula os módulos dos sensores
    modules = calculate_sensor_modules(data)
    
    # Mapeamento de atividades (baseado na descrição do dataset)
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
    
    # Cria a figura com subplots organizados
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Cores para cada dispositivo
    device_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Para cada tipo de sensor (3 linhas)
    sensor_types = ['acc_module', 'gyro_module', 'mag_module']
    sensor_titles = ['Módulo do Acelerómetro', 'Módulo do Giroscópio', 'Módulo do Magnetómetro']
    
    for sensor_idx, (sensor_type, sensor_title) in enumerate(zip(sensor_types, sensor_titles)):
        for device_id in range(1, 6):  # 5 dispositivos (5 colunas)
            ax = axes[sensor_idx, device_id - 1]
            
            # Filtra dados para o dispositivo específico
            device_data = data[data[:, 0] == device_id]
            device_modules = modules[sensor_type][data[:, 0] == device_id]
            device_activities = device_data[:, 11]  # Coluna 12: Activity Label
            
            # Prepara dados para boxplot (agrupa por atividade)
            boxplot_data = []
            activity_labels = []
            
            for activity_id in sorted(np.unique(device_activities)):
                activity_mask = device_activities == activity_id
                activity_modules = device_modules[activity_mask]
                
                if len(activity_modules) > 0:  # Só inclui se houver dados
                    boxplot_data.append(activity_modules)
                    activity_labels.append(f"{int(activity_id)}")
            
            # Cria o boxplot
            if boxplot_data:
                bp = ax.boxplot(boxplot_data, patch_artist=True, labels=activity_labels)
                
                # Colore as caixas
                for patch in bp['boxes']:
                    patch.set_facecolor(device_colors[device_id - 1])
                    patch.set_alpha(0.7)
                
                # Estiliza os elementos
                for element in ['whiskers', 'fliers', 'medians', 'caps']:
                    plt.setp(bp[element], color='black', linewidth=1)
                
                # Configura o eixo
                ax.set_title(f'{device_names[device_id]}', fontsize=10, fontweight='bold')
                ax.set_xlabel('Atividade', fontsize=8)
                ax.set_ylabel('Módulo', fontsize=8)
                ax.tick_params(axis='x', rotation=45, labelsize=7)
                ax.tick_params(axis='y', labelsize=7)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Sem dados', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{device_names[device_id]}', fontsize=10, fontweight='bold')
        
        # Adiciona título da linha (tipo de sensor)
        axes[sensor_idx, 0].text(-0.2, 0.5, sensor_title, rotation=90, 
                                ha='center', va='center', transform=axes[sensor_idx, 0].transAxes,
                                fontsize=12, fontweight='bold')
    
    # Ajusta o layout para evitar sobreposição
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, right=0.95, top=0.93, bottom=0.15)
    
    # Salva a figura diretamente na pasta plots
    import os
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    if participant_id == "todos_participantes":
        filename = 'boxplot_todos_participantes.png'
    else:
        filename = f'boxplot_participant_{participant_id}.png'
    
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()  # Fecha a figura para liberar memória
    
    print(f"Boxplot salvo em: {filepath}")
    return fig

