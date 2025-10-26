"""
Script principal do projeto ECAC - Engenharia de Características
Executa todos os exercícios de análise de dados de sensores e deteção de outliers.
"""

import sys
import os
import numpy as np
import time

# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.modules.data_loader import load_participant_data, load_all_participants_data
from src.modules.boxplot_visualization import create_boxplot_visualization
from src.modules.outlier_density_analysis import calculate_outlier_density, analyze_outlier_patterns
from src.modules.zscore_outlier_detection import detect_outliers_zscore, create_zscore_plots, compare_methods
from src.modules.kmeans_outlier_detection import analyze_kmeans_outliers, compare_with_zscore
from src.modules.dbscan_outlier_detection import analyze_dbscan_outliers, compare_dbscan_with_kmeans, summarize_dbscan_analysis

def format_time(seconds):
    """
    Converte segundos para formato legível (s, m:s, ou h:m:s).
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.2f}s"

def main():
    """
    Executa todos os exercícios do projeto em sequência.
    Inclui carregamento de dados, análise de outliers e visualizações.
    """
    print("=" * 60)
    print("PROJETO ECAC - ENGENHARIA DE CARACTERÍSTICAS")
    print("=" * 60)
    
    # Dicionário para armazenar tempos de execução
    execution_times = {}
    
    try:
        # EXERCÍCIO 2: Data Loading - Participante Específico
        print(f"\nEXERCÍCIO 2: DATA LOADING - PARTICIPANTE ESPECÍFICO")
        print("-" * 60)
        start_time = time.time()
        
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
        
        execution_times['Exercício 2'] = time.time() - start_time
        print(f"\nTempo de execução: {format_time(execution_times['Exercício 2'])}")
        print("Exercício 2 concluído!")
        
        # CARREGAR DADOS DE TODOS OS PARTICIPANTES para exercícios seguintes
        print(f"\nCARREGANDO TODOS OS PARTICIPANTES PARA ANÁLISE GLOBAL")
        print("-" * 60)
        start_time = time.time()
        all_data, participant_info = load_all_participants_data()
        load_time = time.time() - start_time
        print(f"Tempo de carregamento: {format_time(load_time)}")
        
        # EXERCÍCIO 3.1: Boxplot Visualization - TODOS OS PARTICIPANTES
        print(f"\nEXERCÍCIO 3.1: BOXPLOT VISUALIZATION - TODOS OS PARTICIPANTES")
        print("-" * 60)
        start_time = time.time()
        
        print("Analisando módulos dos sensores (acelerómetro, giroscópio, magnetómetro)")
        print("Combinando dados de todos os participantes")
        print("Separados por atividade e dispositivo")
        
        print("\nCriando boxplots organizados em grid...")
        # Cria pasta para este exercício
        output_dir_31 = "plots/exercicio_3.1_boxplot"
        os.makedirs(output_dir_31, exist_ok=True)
        create_boxplot_visualization(all_data, "todos_participantes", output_dir=output_dir_31)
        
        execution_times['Exercício 3.1'] = time.time() - start_time
        print(f"\nTempo de execução (incluindo gráficos): {format_time(execution_times['Exercício 3.1'])}")
        print("Exercício 3.1 concluído!")
        
        # EXERCÍCIO 3.2: Outlier Density Analysis - TODOS OS PARTICIPANTES
        print(f"\nEXERCÍCIO 3.2: OUTLIER DENSITY ANALYSIS - TODOS OS PARTICIPANTES")
        print("-" * 60)
        start_time = time.time()
        
        print("Analisando densidade de outliers usando método IQR (Tukey)")
        print("Focando apenas nos sensores do pulso direito")
        print("Dados combinados de todos os participantes")
        
        print("\nCalculando densidades de outliers...")
        # Cria pasta para este exercício
        output_dir_32 = "plots/exercicio_3.2_outlier_density"
        os.makedirs(output_dir_32, exist_ok=True)
        results = calculate_outlier_density(all_data, "todos_participantes", output_dir=output_dir_32)
        
        print("\nAnalisando padrões...")
        analyze_outlier_patterns(results)
        
        execution_times['Exercício 3.2'] = time.time() - start_time
        print(f"\nTempo de execução (incluindo gráficos): {format_time(execution_times['Exercício 3.2'])}")
        print("Exercício 3.2 concluído!")
        
        # EXERCÍCIO 3.3: Implementação da função Z-Score
        print(f"\nEXERCÍCIO 3.3: IMPLEMENTAÇÃO DA FUNÇÃO Z-SCORE")
        print("-" * 60)
        start_time = time.time()
        print("Função detect_outliers_zscore(data, k) implementada!")
        print("Esta função:")
        print("  • Recebe um array de amostras e um parâmetro k")
        print("  • Calcula Z-Score: Z = (x - μ) / σ")
        print("  • Identifica outliers onde |Z| > k")
        print("  • Retorna array booleano de outliers")
        
        execution_times['Exercício 3.3'] = time.time() - start_time
        print(f"\nTempo de execução: {format_time(execution_times['Exercício 3.3'])}")
        print("Exercício 3.3 concluído!")
        
        # EXERCÍCIO 3.4: Plots com Z-Score para diferentes valores de k
        print(f"\nEXERCÍCIO 3.4: DETEÇÃO DE OUTLIERS COM Z-SCORE")
        print("-" * 60)
        start_time = time.time()
        print("Criando plots separados por atividade e dispositivo")
        print("Outliers em VERMELHO, pontos normais em AZUL")
        print("Testando com k = 3, 3.5 e 4")
        
        print("\nGerando gráficos (isto pode demorar alguns segundos)...")
        # Cria pasta para este exercício com subpastas por k
        output_dir_34 = "plots/exercicio_3.4_zscore"
        os.makedirs(output_dir_34, exist_ok=True)
        create_zscore_plots(all_data, k_values=[3, 3.5, 4], output_dir=output_dir_34)
        
        execution_times['Exercício 3.4'] = time.time() - start_time
        print(f"\nTempo de execução (incluindo gráficos): {format_time(execution_times['Exercício 3.4'])}")
        print("Exercício 3.4 concluído!")
        
        # EXERCÍCIO 3.5: Comparação entre métodos IQR e Z-Score
        print(f"\nEXERCÍCIO 3.5: COMPARAÇÃO IQR vs Z-SCORE")
        print("-" * 60)
        start_time = time.time()
        print("Comparando métodos para sensores do pulso direito")
        print("Analisando densidades de outliers obtidas com cada método")
        
        # Cria pasta para este exercício
        output_dir_35 = "plots/exercicio_3.5_comparacao"
        os.makedirs(output_dir_35, exist_ok=True)
        compare_methods(all_data, output_dir=output_dir_35)
        
        execution_times['Exercício 3.5'] = time.time() - start_time
        print(f"\nTempo de execução (incluindo gráficos): {format_time(execution_times['Exercício 3.5'])}")
        print("\nExercício 3.5 concluído!")
        
        # EXERCÍCIO 3.6 e 3.7: K-Means Clustering para deteção de outliers
        print(f"\nEXERCÍCIO 3.6 e 3.7: K-MEANS CLUSTERING PARA DETEÇÃO DE OUTLIERS")
        print("-" * 60)
        start_time = time.time()
        print("Implementando algoritmo K-Means (ex 3.6)")
        print("Aplicando K-Means para detectar outliers (ex 3.7)")
        print("Testando diferentes números de clusters: k = 3, 5, 7")
        print("Usando espaço dos módulos dos sensores (3D)")
        print("Usando amostra de 1/5 dos dados para eficiência")
        
        print("\nExecutando análise K-Means...")
        # Cria pasta para este exercício com subpastas
        output_dir_36 = "plots/exercicio_3.6_3.7_kmeans"
        output_dir_normal = os.path.join(output_dir_36, "normal")
        output_dir_zoom = os.path.join(output_dir_36, "zoom")
        os.makedirs(output_dir_normal, exist_ok=True)
        os.makedirs(output_dir_zoom, exist_ok=True)
        
        kmeans_results = analyze_kmeans_outliers(
            all_data, 
            cluster_range=[3, 5, 7],
            use_modules=True,
            create_plots=True,
            output_dir_normal=output_dir_normal,
            output_dir_zoom=output_dir_zoom
        )
        
        print("\nComparando resultados K-Means com Z-Score...")
        compare_with_zscore(all_data, kmeans_results, k_zscore=3)
        
        execution_times['Exercícios 3.6 e 3.7'] = time.time() - start_time
        print(f"\n⏱️  Tempo de execução (incluindo gráficos): {format_time(execution_times['Exercícios 3.6 e 3.7'])}")
        print("\nExercícios 3.6 e 3.7 concluídos!")
        
        # EXERCÍCIO 3.7.1: DBSCAN para deteção de outliers
        print(f"\nEXERCÍCIO 3.7.1: DBSCAN PARA DETEÇÃO DE OUTLIERS")
        print("-" * 60)
        start_time = time.time()
        
        print("Implementando análise com DBSCAN (usando sklearn)")
        print("Testando apenas as 2 primeiras combinações de parâmetros:")
        print("  • eps (epsilon): raio da vizinhança")
        print("  • min_samples: mínimo de pontos para formar cluster")
        print("Usando AMOSTRA dos dados para processamento mais rápido")
        
        # Calcular tamanho da amostra: 1/25 dos dados
        dbscan_sample_size = len(all_data) // 25
        print(f"\nExecutando análise DBSCAN com amostra de {dbscan_sample_size:,} pontos (1/25 de {len(all_data):,} totais)...")
        print(f"Configurações: (eps=0.5, ms=5), (eps=0.8, ms=5)")
        print(f"  → Otimizado para melhor performance!")
        
        # Cria pasta para este exercício
        output_dir_371 = "plots/exercicio_3.7.1_dbscan"
        os.makedirs(output_dir_371, exist_ok=True)
        
        # Executar apenas as 2 combinações específicas
        dbscan_results = {}
        configs = [(0.5, 5), (0.8, 5)]
        
        for eps, min_samples in configs:
            from src.modules.dbscan_outlier_detection import detect_outliers_dbscan, create_3d_visualization_dbscan, sample_data
            
            key = f"eps{eps}_ms{min_samples}"
            print(f"\nTestando eps={eps}, min_samples={min_samples}...")
            
            # Amostrar dados
            data_sample = sample_data(all_data, sample_size=dbscan_sample_size)
            
            # Detectar outliers
            result = detect_outliers_dbscan(
                data_sample,
                eps=eps,
                min_samples=min_samples,
                use_modules=True,
                normalize=True
            )
            
            dbscan_results[key] = result
            
            print(f"  ✓ {result['n_clusters']} clusters encontrados")
            print(f"  ✓ {result['n_outliers']:,} outliers ({result['outlier_percentage']:.2f}%)")
            
            # Criar visualização
            print(f"  Criando visualização 3D...")
            fig = create_3d_visualization_dbscan(
                data_sample,
                result,
                title_suffix=f"Amostra: {len(data_sample):,} pontos"
            )
            
            filename = f"dbscan_3d_eps{eps}_ms{min_samples}.png"
            filepath = os.path.join(output_dir_371, filename)
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            import matplotlib.pyplot as plt
            plt.close(fig)
            print(f"  ✓ Gráfico salvo: {filepath}")
        
        # Resumo da análise
        summarize_dbscan_analysis(dbscan_results)
        
        # Comparar DBSCAN com K-Means (usando eps=0.8, min_samples=5)
        if 'eps0.8_ms5' in dbscan_results and kmeans_results:
            # Pegar resultado K-Means com k=5 para comparação
            kmeans_k5 = kmeans_results.get('k=5', list(kmeans_results.values())[0])
            print(f"\nComparando DBSCAN (eps=0.8, ms=5) com K-Means (k=5)...")
            
            # Nota: Os dados da comparação são diferentes (amostra vs completo)
            # então a comparação é apenas ilustrativa
            print("\nNOTA: DBSCAN usa amostra menor (1/25), K-Means usa amostra maior (1/5)")
            print("Comparação é apenas ilustrativa dos métodos")
        
        execution_times['Exercício 3.7.1'] = time.time() - start_time
        print(f"\n⏱️  Tempo de execução (incluindo gráficos): {format_time(execution_times['Exercício 3.7.1'])}")
        print("\nExercício 3.7.1 concluído!")
        
        # Resumo de tempos de execução
        print(f"\n{'=' * 60}")
        print(f"RESUMO DE TEMPOS DE EXECUÇÃO")
        print(f"{'=' * 60}")
        for exercise, exec_time in execution_times.items():
            print(f"{exercise:30s}: {format_time(exec_time)}")
        print(f"{'=' * 60}")
        
        print(f"\nPROJETO CONCLUÍDO!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()
