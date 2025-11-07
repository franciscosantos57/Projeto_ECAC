"""
Funções para carregar e processar dados do dataset de sensores.
Carrega dados de participantes individuais ou todos os participantes combinados.
"""

import numpy as np
import os


def load_participant_data(participant_id, dataset_path="dataset"):
    """
    Carrega dados de um participante específico de todos os 5 dispositivos.
    
    Args:
        participant_id (int): ID do participante (0-14)
        dataset_path (str): Caminho para a pasta do dataset
    
    Returns:
        numpy.ndarray: Dados combinados [n_samples, 12]
    """
    participant_folder = f"part{participant_id}"
    participant_path = os.path.join(dataset_path, participant_folder)
    
    if not os.path.exists(participant_path):
        raise FileNotFoundError(f"Pasta do participante {participant_id} não encontrada: {participant_path}")
    
    device_arrays = []
    
    # Carrega dados de todos os dispositivos (1-5)
    for device_id in range(1, 6):
        filename = f"part{participant_id}dev{device_id}.csv"
        filepath = os.path.join(participant_path, filename)
        
        if not os.path.exists(filepath):
            print(f"Aviso: Ficheiro {filename} não encontrado")
            continue
        
        # Carrega dados do CSV
        device_data = np.loadtxt(filepath, delimiter=',', dtype=np.float64)
        device_arrays.append(device_data)
    
    # Combina todos os dispositivos num único array
    if device_arrays:
        return np.vstack(device_arrays)
    else:
        return np.array([]).reshape(0, 12)

def load_all_participants_data(dataset_path="dataset"):
    """
    Carrega dados de todos os participantes (0-14) numa única matriz.
    
    Returns:
        tuple: (dados_combinados, info_participantes)
            - dados_combinados: Array com todos os dados
            - info_participantes: Dict com contagem de amostras por participante
    """
    print("Carregando dados de todos os participantes...")
    all_data = []
    participant_info = {}
    
    # Carrega dados de todos os participantes (0-14)
    for participant_id in range(15):
        try:
            participant_data = load_participant_data(participant_id, dataset_path)
            if len(participant_data) > 0:
                all_data.append(participant_data)
                participant_info[participant_id] = len(participant_data)
                print(f"  Participante {participant_id}: {len(participant_data)} amostras")
            else:
                print(f"  Participante {participant_id}: Sem dados")
                participant_info[participant_id] = 0
        except Exception as e:
            print(f"  Erro ao carregar participante {participant_id}: {e}")
            participant_info[participant_id] = 0
    
    # Combina todos os dados
    if all_data:
        combined_data = np.vstack(all_data)
        print(f"\nDados combinados: {len(combined_data)} amostras totais de {len([p for p in participant_info.values() if p > 0])} participantes")
        return combined_data, participant_info
    else:
        return np.array([]).reshape(0, 12), participant_info
