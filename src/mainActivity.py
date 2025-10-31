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
from src.modules.dbscan_outlier_detection import analyze_dbscan_outliers, summarize_dbscan_analysis
from src.modules.statistical_significance import analyze_statistical_significance
from src.modules.feature_extraction import extract_features_from_windows
from src.modules.feature_analysis import analyze_feature_set, create_feature_visualizations, save_feature_set
from src.modules.pca_analysis import (normalize_features_zscore, apply_pca, analyze_variance_explained,
                                      create_variance_plot, print_pca_analysis, print_component_contributions,
                                      example_feature_compression, print_compression_example)
from src.modules.feature_comparison import compare_feature_selection_methods
from src.utils.sliding_windows import create_sliding_windows, get_window_statistics


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
        print(f"Dados carregados: {single_participant_data.shape[0]} amostras, {single_participant_data.shape[1]} colunas")
        
        # Mostra estrutura dos dados
        print(f"\nMatriz de dados do participante {participant_id}:")
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
        print("Usando amostra de 1/10 dos dados para eficiência")
        
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
        print(f"\nTempo de execução (incluindo gráficos): {format_time(execution_times['Exercícios 3.6 e 3.7'])}")
        print("\nExercícios 3.6 e 3.7 concluídos!")
        
        # EXERCÍCIO 3.7.1: DBSCAN para deteção de outliers
        print(f"\nEXERCÍCIO 3.7.1: DBSCAN PARA DETEÇÃO DE OUTLIERS")
        print("-" * 60)
        start_time = time.time()
        
        print("Implementando análise com DBSCAN (usando sklearn)")
        print("Testando apenas as 2 primeiras combinações de parâmetros:")
        print("  • eps (epsilon): raio da vizinhança")
        print("  • min_samples: mínimo de pontos para formar cluster")
        print("NOTA: DBSCAN usa 1/50 dos dados para evitar problemas de memória")
        
        # Usar 1/50 dos dados (DBSCAN é muito pesado em memória)
        dbscan_sample_size = len(all_data) // 50
        print(f"\nExecutando análise DBSCAN com amostra de {dbscan_sample_size:,} pontos (1/50 de {len(all_data):,} totais)...")
        print(f"Configurações: (eps=0.5, ms=5), (eps=0.8, ms=5)")
        print(f"(DBSCAN constrói matriz de distâncias N×N, logo requer muito menos pontos que K-Means)")
        
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
            
            print(f"  {result['n_clusters']} clusters encontrados")
            print(f"  {result['n_outliers']:,} outliers ({result['outlier_percentage']:.2f}%)")
            
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
            print(f"  Gráfico salvo: {filepath}")
        
        # Resumo da análise
        summarize_dbscan_analysis(dbscan_results)
        
        # Comparação com K-Means (informativa)
        if kmeans_results:
            print(f"\nNOTA COMPARATIVA:")
            print(f"  • DBSCAN: amostra 1/100 (~39k pontos), deteta clusters de forma arbitrária")
            print(f"  • K-Means: amostra 1/25 (~157k pontos), assume clusters esféricos")
            print(f"  • Métodos complementares para análise de outliers")
        
        execution_times['Exercício 3.7.1'] = time.time() - start_time
        print(f"\nTempo de execução (incluindo gráficos): {format_time(execution_times['Exercício 3.7.1'])}")
        print("\nExercício 3.7.1 concluído!")
        
        # EXERCÍCIO 4.1: Análise de Significância Estatística
        print(f"\nEXERCÍCIO 4.1: ANÁLISE DE SIGNIFICÂNCIA ESTATÍSTICA")
        print("-" * 60)
        start_time = time.time()
        
        print("Testando normalidade das distribuições (Kolmogorov-Smirnov)")
        print("Aplicando testes de significância (ANOVA ou Kruskal-Wallis)")
        print("Determinando poder discriminante dos módulos dos sensores")
        
        # Executa análise de significância (sem gráficos)
        significance_results = analyze_statistical_significance(all_data, output_dir=None)
        
        execution_times['Exercício 4.1'] = time.time() - start_time
        print(f"\nTempo de execução: {format_time(execution_times['Exercício 4.1'])}")
        print("\nExercício 4.1 concluído!")
        
        # EXERCÍCIO 4.2: Extração de Features Temporais e Espectrais
        print(f"\nEXERCÍCIO 4.2: EXTRAÇÃO DE FEATURES TEMPORAIS E ESPECTRAIS")
        print("-" * 60)
        start_time = time.time()
        
        print("Baseado no artigo de Zhang & Sawchuk")
        print("Implementando sliding windows (5s, overlap 50%)")
        print("Extraindo features temporais e espectrais")
        
        # Parâmetros de segmentação
        window_size_sec = 5
        overlap = 0.5
        sampling_rate = 50  # Hz (baseado no dataset)
        
        print(f"\nParâmetros de segmentação:")
        print(f"  • Tamanho da janela: {window_size_sec}s")
        print(f"  • Overlap: {overlap * 100}%")
        print(f"  • Taxa de amostragem: {sampling_rate} Hz")
        print(f"  • Amostras por janela: {window_size_sec * sampling_rate}")
        
        # Cria sliding windows
        print(f"\nCriando sliding windows...")
        windows = create_sliding_windows(all_data, 
                                        window_size_sec=window_size_sec,
                                        overlap=overlap,
                                        sampling_rate=sampling_rate)
        
        # Estatísticas das janelas
        window_stats = get_window_statistics(windows)
        print(f"\nEstatísticas das janelas:")
        print(f"  • Total de janelas: {window_stats['total_windows']}")
        print(f"  • Janelas válidas: {window_stats['valid_windows']}")
        print(f"  • Janelas descartadas: {window_stats['discarded_windows']} ({window_stats['discard_rate']:.2f}%)")
        
        # Extrai features
        print(f"\nExtraindo features temporais e espectrais...")
        feature_matrix, labels, metadata, feature_names = extract_features_from_windows(
            windows, 
            sampling_rate=sampling_rate,
            verbose=True
        )
        
        # Análise do feature set
        analyze_feature_set(feature_matrix, labels, metadata, feature_names)
        
        # Cria visualizações
        output_dir_42 = "plots/exercicio_4.2_features"
        create_feature_visualizations(feature_matrix, labels, metadata, feature_names, 
                                     output_dir=output_dir_42)
        
        # Salva feature set
        save_feature_set(feature_matrix, labels, metadata, feature_names, 
                        output_dir="data/features")
        
        execution_times['Exercício 4.2'] = time.time() - start_time
        print(f"\nTempo de execução: {format_time(execution_times['Exercício 4.2'])}")
        print("\nExercício 4.2 concluído!")
        
        # EXERCÍCIO 4.3 e 4.4: PCA para Redução de Dimensionalidade
        print(f"\nEXERCÍCIO 4.3 e 4.4: PCA PARA REDUÇÃO DE DIMENSIONALIDADE")
        print("-" * 60)
        start_time = time.time()
        
        print("Aplicando PCA para comprimir o espaço de features")
        
        # Normalizar features com Z-Score
        X_normalized, scaler = normalize_features_zscore(feature_matrix)
        
        # Aplicar PCA com todos os componentes
        pca_full, X_transformed_full = apply_pca(X_normalized, n_components=None)
        
        # Analisar variância
        variance_info = analyze_variance_explained(pca_full)
        n_components_75 = variance_info['n_components_for_threshold'][0.75]
        
        # Imprimir análise
        print_pca_analysis(variance_info, feature_matrix.shape[1])
        
        # Criar gráfico de variância
        output_dir_43 = "plots/exercicio_4.3_pca"
        plot_path = create_variance_plot(pca_full, variance_info, output_dir=output_dir_43)
        print(f"Gráfico salvo: {plot_path}")
        
        # Exemplo de compressão
        example_data = example_feature_compression(feature_matrix, labels, n_components_75)
        print_compression_example(example_data)
        
        # Análise de contribuições
        print_component_contributions(pca_full)
        
        execution_times['Exercícios 4.3 e 4.4'] = time.time() - start_time
        print(f"\nTempo de execução: {format_time(execution_times['Exercícios 4.3 e 4.4'])}")
        print("Exercícios 4.3 e 4.4 concluídos!")
        
        # EXERCÍCIO 4.5 e 4.6: Seleção de Features com Fisher Score e ReliefF
        print(f"\nEXERCÍCIO 4.5 e 4.6: SELEÇÃO DE FEATURES (FISHER SCORE E ReliefF)")
        print("-" * 60)
        start_time = time.time()
        
        print("Identificando top-10 features com Fisher Score e ReliefF")
        
        # Comparar métodos de seleção de features
        feature_selection_results = compare_feature_selection_methods(
            X=feature_matrix,
            y=labels,
            feature_names=feature_names,
            top_k=10,
            relieff_neighbors=10,
            relieff_samples=100,
            verbose=True
        )
        
        execution_times['Exercícios 4.5 e 4.6'] = time.time() - start_time
        print(f"\nTempo de execução: {format_time(execution_times['Exercícios 4.5 e 4.6'])}")
        print("Exercícios 4.5 e 4.6 concluídos!")
        
        # EXERCÍCIO 4.6.1: Extração de Features Selecionadas
        print(f"\nEXERCÍCIO 4.6.1: EXTRAÇÃO DE FEATURES SELECIONADAS")
        print("-" * 60)
        start_time = time.time()
        
        from src.modules.feature_comparison import demonstrate_feature_extraction
        
        # Demonstra extração para um instante aleatório
        sample_idx = np.random.randint(0, len(feature_matrix))
        extraction_example = demonstrate_feature_extraction(
            X=feature_matrix,
            feature_names=feature_names,
            fisher_ranking=feature_selection_results['fisher_ranking'],
            relieff_ranking=feature_selection_results['relieff_ranking'],
            sample_idx=sample_idx,
            verbose=True
        )
        
        execution_times['Exercício 4.6.1'] = time.time() - start_time
        print(f"\nTempo de execução: {format_time(execution_times['Exercício 4.6.1'])}")
        print("Exercício 4.6.1 concluído!")
        
        # EXERCÍCIO 4.6.2: Análise de Vantagens e Limitações
        print(f"\nEXERCÍCIO 4.6.2: ANÁLISE DE VANTAGENS E LIMITAÇÕES")
        print("-" * 60)
        start_time = time.time()
        
        from src.modules.feature_comparison import analyze_selection_approach
        
        # Analisa vantagens e limitações
        approach_analysis = analyze_selection_approach(
            fisher_ranking=feature_selection_results['fisher_ranking'],
            relieff_ranking=feature_selection_results['relieff_ranking'],
            X_shape=feature_matrix.shape,
            verbose=True
        )
        
        execution_times['Exercício 4.6.2'] = time.time() - start_time
        print(f"\nTempo de execução: {format_time(execution_times['Exercício 4.6.2'])}")
        print("Exercício 4.6.2 concluído!")
        print("\nAnálise detalhada disponível em ANALISES_E_CONCLUSOES.txt")
        
        # Resumo de tempos de execução
        print(f"\n{'=' * 60}")
        print(f"RESUMO DE TEMPOS DE EXECUÇÃO")
        print(f"{'=' * 60}")
        total_time = 0
        for exercise, exec_time in execution_times.items():
            print(f"{exercise:30s}: {format_time(exec_time)}")
            total_time += exec_time
        print(f"{'=' * 60}")
        print(f"{'TEMPO TOTAL':30s}: {format_time(total_time)}")
        print(f"{'=' * 60}")
        
        print(f"\nPROJETO CONCLUÍDO!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()
