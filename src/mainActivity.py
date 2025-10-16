import sys
import os
import numpy as np

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.utils.data_loader import load_participant_data, load_all_participants_data
from src.utils.boxplot_visualization import create_boxplot_visualization
from src.utils.outlier_density_analysis import calculate_outlier_density, analyze_outlier_patterns

def main():
    """
    Função principal para executar os exercícios do projeto
    """
    print("=" * 60)
    print("PROJETO ECAC - ENGENHARIA DE CARACTERÍSTICAS")
    print("=" * 60)
    
    try:
        # EXERCÍCIO 2: Data Loading - Participante Específico
        print(f"\nEXERCÍCIO 2: DATA LOADING - PARTICIPANTE ESPECÍFICO")
        print("-" * 60)
        participant_id = 0
        print(f"Carregando dados do participante {participant_id}...")
        
        single_participant_data = load_participant_data(participant_id)
        print(f"✓ Dados carregados: {single_participant_data.shape[0]} amostras, {single_participant_data.shape[1]} colunas")
        
        # Mostrar a matriz de dados do participante
        print(f"\n✓ Matriz de dados do participante {participant_id}:")
        print("-" * 80)
        print("Formato: [Dev_ID, Acc_X, Acc_Y, Acc_Z, Gyro_X, Gyro_Y, Gyro_Z, Mag_X, Mag_Y, Mag_Z, Timestamp, Activity]")
        print("-" * 80)
        
        # Mostrar primeiras 10 linhas da matriz com formatação melhorada
        print("Primeiras 10 amostras:")
        print(f"{'#':<3} {'Dev':<3} {'Acc_X':<8} {'Acc_Y':<8} {'Acc_Z':<8} {'Gyro_X':<8} {'Gyro_Y':<8} {'Gyro_Z':<8} {'Mag_X':<8} {'Mag_Y':<8} {'Mag_Z':<8} {'Time':<8} {'Act':<3}")
        print("-" * 80)
        
        for i in range(min(10, len(single_participant_data))):
            row = single_participant_data[i]
            print(f"{i+1:<3} {int(row[0]):<3} {row[1]:<8.3f} {row[2]:<8.3f} {row[3]:<8.3f} {row[4]:<8.3f} {row[5]:<8.3f} {row[6]:<8.3f} {row[7]:<8.3f} {row[8]:<8.3f} {row[9]:<8.3f} {row[10]:<8.0f} {int(row[11]):<3}")
        
        if len(single_participant_data) > 10:
            print(f"... (mais {len(single_participant_data) - 10} amostras)")
        print("-" * 80)
        
        print("Exercício 2 concluído!")
        
        # CARREGAR DADOS DE TODOS OS PARTICIPANTES para exercícios seguintes
        print(f"\nCARREGANDO TODOS OS PARTICIPANTES PARA ANÁLISE GLOBAL")
        print("-" * 60)
        all_data, participant_info = load_all_participants_data()
        
        # EXERCÍCIO 3.1: Boxplot Visualization - TODOS OS PARTICIPANTES
        print(f"\nEXERCÍCIO 3.1: BOXPLOT VISUALIZATION - TODOS OS PARTICIPANTES")
        print("-" * 60)
        print("Analisando módulos dos sensores (acelerómetro, giroscópio, magnetómetro)")
        print("Combinando dados de todos os participantes")
        print("Separados por atividade e dispositivo")
        
        print("\nCriando boxplots organizados em grid...")
        create_boxplot_visualization(all_data, "todos_participantes")
        
        print("Exercício 3.1 concluído!")
        
        # EXERCÍCIO 3.2: Outlier Density Analysis - TODOS OS PARTICIPANTES
        print(f"\nEXERCÍCIO 3.2: OUTLIER DENSITY ANALYSIS - TODOS OS PARTICIPANTES")
        print("-" * 60)
        print("Analisando densidade de outliers usando método IQR (Tukey)")
        print("Focando apenas nos sensores do pulso direito")
        print("Dados combinados de todos os participantes")
        
        print("\nCalculando densidades de outliers...")
        results = calculate_outlier_density(all_data, "todos_participantes")
        
        print("\nAnalisando padrões...")
        analyze_outlier_patterns(results)
        
        print("Exercício 3.2 concluído!")
        
        print(f"\nPROJETO CONCLUÍDO!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()
