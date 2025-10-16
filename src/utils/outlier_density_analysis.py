import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sem interface gráfica
import matplotlib.pyplot as plt
import os

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

def detect_outliers_iqr(data):
    """
    Detecta outliers usando o método IQR (definição de Tukey).
    
    Args:
        data (numpy.ndarray): Array de dados
    
    Returns:
        numpy.ndarray: Array booleano indicando outliers (True) e não-outliers (False)
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    # Limites para outliers (definição de Tukey)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identifica outliers
    outliers = (data < lower_bound) | (data > upper_bound)
    
    return outliers

def calculate_outlier_density(data, participant_id="todos_participantes"):
    """
    Calcula a densidade de outliers para cada atividade usando apenas sensores do pulso direito.
    
    Args:
        data (numpy.ndarray): Dados carregados (pode ser de um participante ou todos)
        participant_id (str/int): ID do participante ou "todos_participantes" (para salvamento)
    
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
    
    # Calcula estatísticas gerais
    print("\n" + "=" * 60)
    print("ESTATÍSTICAS GERAIS")
    print("-" * 60)
    print(f"Densidade média - Acelerómetro: {np.mean(results['acc_densities']):.2f}%")
    print(f"Densidade média - Giroscópio: {np.mean(results['gyro_densities']):.2f}%")
    print(f"Densidade média - Magnetómetro: {np.mean(results['mag_densities']):.2f}%")
    
    # Identifica atividades com mais outliers
    print(f"\nAtividade com mais outliers - Acelerómetro: {results['activity_names'][np.argmax(results['acc_densities'])]} ({max(results['acc_densities']):.2f}%)")
    print(f"Atividade com mais outliers - Giroscópio: {results['activity_names'][np.argmax(results['gyro_densities'])]} ({max(results['gyro_densities']):.2f}%)")
    print(f"Atividade com mais outliers - Magnetómetro: {results['activity_names'][np.argmax(results['mag_densities'])]} ({max(results['mag_densities']):.2f}%)")
    
    # Cria visualização
    create_outlier_density_plot(results, participant_id)
    
    return results

def create_outlier_density_plot(results, participant_id):
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
    
    # Salva o gráfico diretamente na pasta plots
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    if participant_id == "todos_participantes":
        filename = 'outlier_density_todos_participantes.png'
    else:
        filename = f'outlier_density_participant_{participant_id}.png'
    
    filepath = os.path.join(plots_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()  # Fecha a figura para liberar memória
    
    print(f"\nGráfico salvo em: {filepath}")

def analyze_outlier_patterns(results):
    """
    Analisa padrões nos outliers e fornece insights.
    
    Args:
        results (dict): Resultados da análise de densidade
    """
    print("\n" + "=" * 60)
    print("ANÁLISE DE PADRÕES")
    print("-" * 60)
    
    # Análise por tipo de atividade
    static_activities = [1, 2, 3]  # Stand, Sit, Sit and Talk
    dynamic_activities = [4, 5, 6, 7]  # Walk, Walk and Talk, Climb Stair, Climb Stair and Talk
    transition_activities = [8, 9, 10, 11, 12, 13, 14, 15, 16]  # Transições
    
    def analyze_activity_group(activity_ids, group_name):
        group_indices = [i for i, act_id in enumerate(results['activities']) if act_id in activity_ids]
        
        if group_indices:
            acc_avg = np.mean([results['acc_densities'][i] for i in group_indices])
            gyro_avg = np.mean([results['gyro_densities'][i] for i in group_indices])
            mag_avg = np.mean([results['mag_densities'][i] for i in group_indices])
            
            print(f"\n{group_name}:")
            print(f"  Acelerómetro: {acc_avg:.2f}%")
            print(f"  Giroscópio: {gyro_avg:.2f}%")
            print(f"  Magnetómetro: {mag_avg:.2f}%")
    
    analyze_activity_group(static_activities, "Atividades Estáticas")
    analyze_activity_group(dynamic_activities, "Atividades Dinâmicas")
    analyze_activity_group(transition_activities, "Atividades de Transição")
    
    # Conclusões
    print(f"\nCONCLUSÕES:")
    print("-" * 30)
    
    # Identifica o sensor com mais outliers
    avg_acc = np.mean(results['acc_densities'])
    avg_gyro = np.mean(results['gyro_densities'])
    avg_mag = np.mean(results['mag_densities'])
    
    if avg_gyro > avg_acc and avg_gyro > avg_mag:
        print("- O giroscópio apresenta a maior densidade média de outliers")
        print("- Isto sugere que as rotações são mais variáveis que as acelerações lineares")
    elif avg_acc > avg_gyro and avg_acc > avg_mag:
        print("- O acelerómetro apresenta a maior densidade média de outliers")
        print("- Isto sugere que as acelerações lineares são mais variáveis")
    else:
        print("- O magnetómetro apresenta a maior densidade média de outliers")
        print("- Isto pode indicar interferências magnéticas ou orientação variável")
    
    print(f"- Densidade média geral: {(avg_acc + avg_gyro + avg_mag)/3:.2f}%")
    print("- Valores baixos (<5%) indicam dados consistentes")
    print("- Valores altos (>10%) podem indicar ruído ou transições complexas")
