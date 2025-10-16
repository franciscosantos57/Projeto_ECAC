import numpy as np
import os
import argparse

def load_participant_data(participant_id, dataset_path="dataset"):
    """
    Carrega os dados de um participante específico.
    
    Args:
        participant_id (int): ID do participante (0-14)
        dataset_path (str): Caminho para a pasta do dataset
    
    Returns:
        numpy.ndarray: Array com os dados do participante
                      Formato: [samples, features]
                      Features: [device_id, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, 
                                mag_x, mag_y, mag_z, timestamp, activity_label]
    """
    participant_folder = f"part{participant_id}"
    participant_path = os.path.join(dataset_path, participant_folder)
    
    if not os.path.exists(participant_path):
        raise FileNotFoundError(f"Pasta do participante {participant_id} não encontrada: {participant_path}")
    
    # Lista para armazenar arrays de cada dispositivo
    device_arrays = []
    
    # Carrega dados de todos os dispositivos (1-5) do participante
    for device_id in range(1, 6):
        filename = f"part{participant_id}dev{device_id}.csv"
        filepath = os.path.join(participant_path, filename)
        
        if not os.path.exists(filepath):
            print(f"Aviso: Ficheiro {filename} não encontrado")
            continue
        
        # Carrega diretamente com numpy
        device_data = np.loadtxt(filepath, delimiter=',', dtype=np.float64)
        device_arrays.append(device_data)
    
    # Concatena todos os arrays de uma vez (O(n) em vez de O(n²))
    if device_arrays:
        return np.vstack(device_arrays)
    else:
        return np.array([]).reshape(0, 12)

def load_all_participants_data(dataset_path="dataset"):
    """
    Carrega os dados de TODOS os participantes numa única matriz.
    
    Args:
        dataset_path (str): Caminho para a pasta do dataset
    
    Returns:
        numpy.ndarray: Array com os dados de todos os participantes
        dict: Informações sobre quantas amostras cada participante contribuiu
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
