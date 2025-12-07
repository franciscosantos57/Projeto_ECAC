"""
Script principal do projeto ECAC - Engenharia de Características
Executa todos os exercícios de análise de dados de sensores e deteção de outliers.
"""

import sys
import os
import numpy as np
import time
from tqdm import tqdm


# Adiciona o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Imports Meta 1
from src.modules.meta1 import (
    load_participant_data, load_all_participants_data,
    create_boxplot_visualization,
    calculate_outlier_density, analyze_outlier_patterns,
    create_zscore_plots, compare_methods,
    analyze_kmeans_outliers, compare_with_zscore,
    analyze_dbscan_outliers, summarize_dbscan_analysis,
    analyze_statistical_significance,
    extract_features_from_windows,
    analyze_feature_set, save_feature_set, load_feature_set,
    normalize_features_zscore, apply_pca, analyze_variance_explained,
    create_variance_plot, print_pca_analysis, print_component_contributions,
    example_feature_compression, print_compression_example,
    compare_feature_selection_methods
)

# Imports Meta 2
from src.modules.meta2 import (
    analyze_dataset_balance, demonstrate_smote, balance_dataset_smote,
    split_within_subject, split_between_subject, compare_splitting_strategies,
    prepare_all_scenarios,
    train_knn, predict_knn,
    calculate_metrics, print_metrics,
    perform_multiple_splits, evaluate_with_multiple_splits,
    print_summary_table, plot_average_confusion_matrix,
    plot_hypothesis_tests, find_best_model,
    run_classification, evaluate_deployment_accuracy
)

# Imports Utils
from src.utils import (
    create_sliding_windows, get_window_statistics,
    cache_exists, load_results, save_results,
    ModelLogger
)


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
    # ========== CONFIGURAÇÕES ==========
    USE_CACHED_FEATURES = True   # Se True, carrega features do cache; se False, recalcula
    USE_SKLEARN_KNN = True       # Se True, usa sklearn k-NN; se False, usa implementação própria
    
    print("=" * 60)
    print("PROJETO ECAC - ENGENHARIA DE CARACTERÍSTICAS PARA APRENDIZAGEM COMPUTACIONAL")
    print("=" * 60)

    # =====================================================================
    # META 1: ENGENHARIA DE CARACTERÍSTICAS
    # =====================================================================
    
    print("\n" + "=" * 60)
    print("META 1 - ENGENHARIA DE CARACTERÍSTICAS")
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
        print("-" * 105)
        print("Formato: [Dev_ID, Acc_X, Acc_Y, Acc_Z, Gyro_X, Gyro_Y, Gyro_Z, Mag_X, Mag_Y, Mag_Z, Timestamp, Activity]")
        print("-" * 105)
        
        # Mostrar primeiras 10 linhas da matriz com formatação melhorada
        print("Primeiras 10 amostras:")
        print(f"{'#':<3} {'Dev':<3} {'Acc_X':<8} {'Acc_Y':<8} {'Acc_Z':<8} {'Gyro_X':<8} {'Gyro_Y':<8} {'Gyro_Z':<8} {'Mag_X':<8} {'Mag_Y':<8} {'Mag_Z':<8} {'Time':<8} {'Act':<3}")
        print("-" * 105)
        
        for i in range(min(10, len(single_participant_data))):
            row = single_participant_data[i]
            print(f"{i+1:<3} {int(row[0]):<3} {row[1]:<8.3f} {row[2]:<8.3f} {row[3]:<8.3f} {row[4]:<8.3f} {row[5]:<8.3f} {row[6]:<8.3f} {row[7]:<8.3f} {row[8]:<8.3f} {row[9]:<8.3f} {row[10]:<8.0f} {int(row[11]):<3}")
        
        if len(single_participant_data) > 10:
            print(f"... (mais {len(single_participant_data) - 10} amostras)")
        print("-" * 105)
        
        execution_times['Exercício 2'] = time.time() - start_time
        print(f"\nTempo de execução: {format_time(execution_times['Exercício 2'])}")
        print("Exercício 2 concluído!")
        
        # Carrega dados de todos os participantes para exercícios seguintes
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
        output_dir_31 = "plots/meta1/exercicio_3.1_boxplot"
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
        output_dir_32 = "plots/meta1/exercicio_3.2_outlier_density"
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
        output_dir_34 = "plots/meta1/exercicio_3.4_zscore"
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
        output_dir_35 = "plots/meta1/exercicio_3.5_comparacao"
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
        output_dir_36 = "plots/meta1/exercicio_3.6_3.7_kmeans"
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
        
        print("\nExecutando análise DBSCAN...")
        
        # Cria pasta para este exercício
        output_dir_371 = "plots/meta1/exercicio_3.7.1_dbscan"
        os.makedirs(output_dir_371, exist_ok=True)
        
        # Executa análise DBSCAN (análogo ao K-Means)
        dbscan_results = analyze_dbscan_outliers(
            all_data,
            eps_values=[0.5, 0.8],
            min_samples_values=[5],
            create_plots=True
        )
        
        # Resumo da análise
        summarize_dbscan_analysis(dbscan_results)
        
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
        
        # Executa análise de significância
        significance_results = analyze_statistical_significance(all_data, output_dir=None)
        
        execution_times['Exercício 4.1'] = time.time() - start_time
        print(f"\nTempo de execução: {format_time(execution_times['Exercício 4.1'])}")
        print("\nExercício 4.1 concluído!")
        
        # EXERCÍCIO 4.2: Extração de Features Temporais e Espectrais
        print(f"\nEXERCÍCIO 4.2: EXTRAÇÃO DE FEATURES TEMPORAIS E ESPECTRAIS")
        print("-" * 60)
        start_time = time.time()
        
        # Tenta carregar features/embeddings de cache
        cached_data = None
        if USE_CACHED_FEATURES:
            cached_data = load_feature_set(output_dir="data/features")
            if cached_data is not None:
                print("Features e embeddings carregados de cache (data/features/)")
                feature_matrix, labels, metadata, feature_names, embeddings = cached_data
                print(f"Features: {feature_matrix.shape[0]} janelas × {feature_matrix.shape[1]} features")
                if embeddings is not None:
                    print(f"Embeddings: {embeddings.shape[0]} janelas × {embeddings.shape[1]} features")
        
        # Se não há cache ou USE_CACHED_FEATURES=False, extrai do zero
        if cached_data is None:
            print("Baseado no artigo de Zhang & Sawchuk")
            print("Implementando sliding windows (5s, overlap 50%)")
            print("Extraindo features temporais e espectrais")
            
            # Parâmetros de segmentação
            window_size_sec = 5
            overlap = 0.5
            sampling_rate = 50
            
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
            
            # Extrai features e embeddings simultaneamente
            feature_matrix, labels, metadata, feature_names, embeddings = extract_features_from_windows(
                windows, 
                sampling_rate=sampling_rate,
                extract_embeddings=True,
                device="cpu",
                batch_size=32
            )
            
            print(f"Features extraidas: {feature_matrix.shape[0]} janelas × {feature_matrix.shape[1]} features")
            if embeddings is not None:
                print(f"Embeddings extraídos: {embeddings.shape[0]} janelas × {embeddings.shape[1]} features")
            
            # Análise do feature set
            analyze_feature_set(feature_matrix, labels, metadata, feature_names)
            
            # Salva feature set e embeddings
            save_feature_set(feature_matrix, labels, metadata, feature_names, 
                            output_dir="data/features", embeddings=embeddings)
            print(f"Dados guardados: data/features/")
        
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
        output_dir_43 = "plots/meta1/exercicio_4.3_pca"
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
        
        from src.modules.meta1.feature_comparison import demonstrate_feature_extraction
        
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
        
        from src.modules.meta1.feature_comparison import analyze_selection_approach
        
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
        
        # =====================================================================
        # META 2: TRANSFER LEARNING E DATA AUGMENTATION
        # =====================================================================
        
        print("\n" + "=" * 60)
        print("META 2 - TRANSFER LEARNING E DATA AUGMENTATION")
        print("=" * 60)
        
        # EXERCÍCIO 1.1: Análise de Balanço do Dataset
        print(f"\nEXERCÍCIO 1.1: ANÁLISE DE BALANÇO DO DATASET")
        print("-" * 60)
        start_time = time.time()
        
        print("Considerando apenas atividades 1 a 7 (conforme especificação)")
        
        # Filtra apenas atividades 1 a 7 em TODOS os dados da Meta 2
        activity_mask = (labels >= 1) & (labels <= 7)
        X_filtered = feature_matrix[activity_mask]
        y_filtered = labels[activity_mask]
        
        # Filtra embeddings também
        embeddings_filtered = embeddings[activity_mask]
        
        # Filtra metadata (que é uma lista)
        metadata_filtered = [metadata[i] for i in range(len(metadata)) if activity_mask[i]]
        
        # Filtra participant_ids para uso posterior
        participant_ids_filtered = np.array([m['participant_id'] for m in metadata_filtered])
        
        print(f"Amostras antes do filtro: {len(labels)}")
        print(f"Amostras após filtro (atividades 1-7): {len(y_filtered)}")
        print(f"  Features: {X_filtered.shape}")
        print(f"  Embeddings: {embeddings_filtered.shape}")
        
        # Analisa balanço do dataset
        balance_results = analyze_dataset_balance(X_filtered, y_filtered, verbose=True)
        
        execution_times['Exercício 1.1'] = time.time() - start_time
        print(f"\nTempo de execução: {format_time(execution_times['Exercício 1.1'])}")
        print("Exercício 1.1 concluído!")
        
        # EXERCÍCIO 1.2 e 1.3: SMOTE para Data Augmentation
        print(f"\nEXERCÍCIO 1.2 e 1.3: SMOTE PARA DATA AUGMENTATION")
        print("-" * 60)
        start_time = time.time()
        
        print("Gerando 3 amostras sintéticas para Atividade 4 do Participante 3")
        print("Usando apenas amostras desse participante")
        print("Visualização 2D com primeiras 2 features")
        
        # Filtra dados do participante 3, atividades 1-7
        participant_id = 3
        participant_mask = np.array([m['participant_id'] == participant_id for m in metadata_filtered])
        X_participant = X_filtered[participant_mask]
        y_participant = y_filtered[participant_mask]
        
        print(f"\nAmostras do participante {participant_id}: {len(y_participant)}")
        
        # Verifica se há amostras suficientes
        if len(y_participant) == 0:
            raise ValueError(f"Nenhuma amostra encontrada para o participante {participant_id}! "
                           f"Execute novamente o mainActivity.py completo para gerar features "
                           f"com participant_id incluído.")
        
        # Demonstra SMOTE
        smote_results = demonstrate_smote(
            X=X_participant,
            y=y_participant,
            participant_id=participant_id,
            target_activity=4,
            k_samples=3,
            n_neighbors=5,
            output_dir="plots/meta2/exercicio_1.3_smote",
            verbose=True,
            feature_names=feature_names
        )
        
        execution_times['Exercícios 1.2 e 1.3'] = time.time() - start_time
        print(f"\nTempo de execução (incluindo gráfico): {format_time(execution_times['Exercícios 1.2 e 1.3'])}")
        print("Exercícios 1.2 e 1.3 concluídos!")
        
        # EXERCÍCIO 2.1: Análise de Embeddings (extraídos no Ex 4.2)
        print(f"\nEXERCÍCIO 2.1: ANÁLISE DE EMBEDDING FEATURES")
        print("-" * 60)
        start_time = time.time()
        
        print("\nUsando embeddings já filtrados (atividades 1-7)...")
        
        print(f"\nComparação com Features Dataset (ambos filtrados 1-7):")
        print(f"  Features:  {X_filtered.shape[0]} segmentos x {X_filtered.shape[1]} features (handcrafted)")
        print(f"  Embeddings: {embeddings_filtered.shape[0]} segmentos x {embeddings_filtered.shape[1]} features (transfer learning)")
        
        execution_times['Exercício 2.1'] = time.time() - start_time
        print(f"\nTempo de execução: {format_time(execution_times['Exercício 2.1'])}")
        print("Exercício 2.1 concluído!")
        
        # EXERCÍCIO 3: Data Splitting Strategies
        print(f"\nEXERCÍCIO 3: DATA SPLITTING STRATEGIES")
        print("-" * 60)
        start_time = time.time()
        
        print("Aplicando splits TVT (60-20-20%) em FEATURES e EMBEDDINGS (atividades 1-7)")
        print("Comparando estratégias within-subject vs between-subject")
        
        # EXERCÍCIO 3.1: Within-Subject Split (FEATURES)
        print(f"\nEXERCÍCIO 3.1: WITHIN-SUBJECT SPLIT")
        print("-" * 60)
        print("Split 60-20-20% dentro de cada participante")
        
        features_within = split_within_subject(
            X=X_filtered,
            y=y_filtered,
            participant_ids=participant_ids_filtered,
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
            random_state=42
        )
        
        embeddings_within = split_within_subject(
            X=embeddings_filtered,
            y=y_filtered,
            participant_ids=participant_ids_filtered,
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
            random_state=42
        )
        
        print(f"✓ Features split: Train={len(features_within['y_train'])}, "
              f"Val={len(features_within['y_val'])}, Test={len(features_within['y_test'])}")
        print(f"✓ Embeddings split: Train={len(embeddings_within['y_train'])}, "
              f"Val={len(embeddings_within['y_val'])}, Test={len(embeddings_within['y_test'])}")
        
        # EXERCÍCIO 3.2: Between-Subject Split (FEATURES)
        print(f"\nEXERCÍCIO 3.2: BETWEEN-SUBJECT SPLIT")
        print("-" * 60)
        print("Split 9-3-3 participantes distintos")
        
        features_between = split_between_subject(
            X=X_filtered,
            y=y_filtered,
            participant_ids=participant_ids_filtered,
            train_size=9,
            val_size=3,
            test_size=3,
            random_state=42
        )
        
        embeddings_between = split_between_subject(
            X=embeddings_filtered,
            y=y_filtered,
            participant_ids=participant_ids_filtered,
            train_size=9,
            val_size=3,
            test_size=3,
            random_state=42
        )
        
        print(f"✓ Features split: Train={len(features_between['y_train'])}, "
              f"Val={len(features_between['y_val'])}, Test={len(features_between['y_test'])}")
        print(f"✓ Embeddings split: Train={len(embeddings_between['y_train'])}, "
              f"Val={len(embeddings_between['y_val'])}, Test={len(embeddings_between['y_test'])}")
        
        # EXERCÍCIO 3.3: Comparação e Discussão
        print(f"\nEXERCÍCIO 3.3: COMPARAÇÃO E DISCUSSÃO")
        print("-" * 60)
        
        compare_splitting_strategies(features_within, features_between, verbose=True)
        
        execution_times['Exercícios 3.1, 3.2 e 3.3'] = time.time() - start_time
        print(f"\nTempo de execução: {format_time(execution_times['Exercícios 3.1, 3.2 e 3.3'])}")
        print("Exercícios 3.1, 3.2 e 3.3 concluídos!")
        
        # EXERCÍCIO 3.4: Preparação de Cenários (All, PCA, ReliefF)
        print(f"\nEXERCÍCIO 3.4: PREPARAÇÃO DE CENÁRIOS DE DATASETS")
        print("-" * 60)
        start_time = time.time()
        
        print("Preparando 3 cenários para FEATURES e EMBEDDINGS:")
        print("  a) All features/embeddings (normalizado)")
        print("  b) PCA-reduced (90% variância)")
        print("  c) ReliefF-selected (top 15 features)")
        print("\nNOTA: PCA e ReliefF são calculados APENAS com dados de treino")
        
        # Prepara cenários para Features (within-subject)
        print("\n1. WITHIN-SUBJECT SPLIT:")
        scenarios_features_within = prepare_all_scenarios(
            split_data=features_within,
            variance_threshold=0.90,
            top_k_features=15,
            verbose=True
        )
        
        scenarios_embeddings_within = prepare_all_scenarios(
            split_data=embeddings_within,
            variance_threshold=0.90,
            top_k_features=15,
            verbose=True
        )
        
        # Prepara cenários para Features (between-subject)
        print("\n2. BETWEEN-SUBJECT SPLIT:")
        scenarios_features_between = prepare_all_scenarios(
            split_data=features_between,
            variance_threshold=0.90,
            top_k_features=15,
            verbose=True
        )
        
        scenarios_embeddings_between = prepare_all_scenarios(
            split_data=embeddings_between,
            variance_threshold=0.90,
            top_k_features=15,
            verbose=True
        )
        
        execution_times['Exercício 3.4'] = time.time() - start_time
        print(f"\nTempo de execução: {format_time(execution_times['Exercício 3.4'])}")
        print("Exercício 3.4 concluído!")
        
        # EXERCÍCIO 4.1 e 4.2: k-NN Classifier e Métricas de Classificação
        print(f"\nEXERCÍCIO 4.1 e 4.2: k-NN CLASSIFIER E MÉTRICAS DE CLASSIFICAÇÃO")
        print("-" * 60)
        start_time = time.time()
        
        print("Implementação própria de k-NN (Exercício 4.1)")
        print("Função de métricas de classificação (Exercício 4.2)")
        print("Testando com k ímpares de 1 a 20")
        
        # Seleciona um cenário para demonstração (Features, Within-Subject, All features)
        print("\nCenário de teste: Features (Within-Subject, All features)")
        scenario = scenarios_features_within['all']
        
        X_train = scenario['X_train']
        y_train = scenario['y_train']
        X_val = scenario['X_val']
        y_val = scenario['y_val']
        X_test = scenario['X_test']
        y_test = scenario['y_test']
        
        print(f"Train: {X_train.shape[0]} amostras, {X_train.shape[1]} features")
        print(f"Val:   {X_val.shape[0]} amostras")
        print(f"Test:  {X_test.shape[0]} amostras")
        
        # Aplicar SMOTE ao treino
        X_train_balanced, y_train_balanced = balance_dataset_smote(X_train, y_train, n_neighbors=5, verbose=True)
        
        # Treinar e avaliar k-NN com diferentes valores de k (ímpares de 1 a 20)
        k_values = list(range(1, 21, 2))  # [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        
        print(f"\n{'─' * 60}")
        print(f"{'k':<5} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print(f"{'─' * 60}")
        
        results = []
        for k in k_values:
            # Treina modelo com dados balanceados
            if USE_SKLEARN_KNN:
                from sklearn.neighbors import KNeighborsClassifier
                model = KNeighborsClassifier(n_neighbors=k, metric='manhattan', weights='distance')
                model.fit(X_train_balanced, y_train_balanced)
                y_pred_val = model.predict(X_val)
            else:
                model = train_knn(X_train_balanced, y_train_balanced, k=k, verbose=False)
                y_pred_val = predict_knn(model, X_val, verbose=False)
            
            # Calcula métricas customizadas (sem imprimir)
            metrics = calculate_metrics(y_true=y_val, y_pred=y_pred_val, average='macro', verbose=False)
            
            # Armazena resultado (incluindo predições e métricas completas)
            results.append({
                'k': k,
                'y_pred': y_pred_val,
                'metrics': metrics,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            })
            
            # Imprime linha da tabela
            print(f"{k:<5} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}")
        
        print(f"{'─' * 60}")
        
        # Demonstração da função de métricas com output completo (exemplo com k=5)
        print(f"\n{'-' * 60}")
        print("DEMONSTRAÇÃO DA FUNÇÃO calculate_metrics() - Exemplo com k=5")
        print(f"{'-' * 60}")
        
        # Usa resultado já calculado para k=5 (índice 2 na lista: [1, 3, 5, ...])
        k5_result = results[2]  # k=5 é o 3º elemento (índice 2)
        print(f"\nk-NN (k={k5_result['k']}) - Validation Set")
        print_metrics(k5_result['metrics'])
        
        execution_times['Exercícios 4.1 e 4.2'] = time.time() - start_time
        print(f"\nTempo de execução: {format_time(execution_times['Exercícios 4.1 e 4.2'])}")
        print("Exercícios 4.1 e 4.2 concluídos!")
        
        # EXERCÍCIO 5: Evaluation - Hyperparameter Tuning e Testes de Hipótese
        print(f"\nEXERCÍCIO 5: HYPERPARAMETER TUNING E TESTES DE HIPÓTESE")
        print("-" * 60)
        start_time = time.time()
        
        print("Avaliando 2 splits × 3 cenários × 2 datasets = 12 configurações\n")
        
        # Define valores de k para tuning
        k_values_tuning = list(range(1, 21, 2))
        
        # Verificar cache
        use_cache = cache_exists()
        if use_cache:
            print("✓ Cache encontrado! Carregando automaticamente...\n")
            distributions_within, distributions_between = load_results()
        else:
            # Inicializar logger
            logger = ModelLogger(output_dir='logs')
            print(f"Logs: {logger.tuning_file} | {logger.final_file}\n")
            
            n_iterations = 10
        
            distributions_within = {'features': {}, 'embeddings': {}}
            distributions_between = {'features': {}, 'embeddings': {}}
        
            configs = [
                ('within', distributions_within, split_within_subject, 
                 {'train_size': 0.6, 'val_size': 0.2, 'test_size': 0.2}),
                ('between', distributions_between, split_between_subject, 
                 {'train_size': 9, 'val_size': 3, 'test_size': 3})
            ]
        
            total_evals = 2 * 2 * 3  # 2 splits × 2 datasets × 3 cenários
            pbar = tqdm(total=total_evals, desc="Avaliando modelos", ncols=80)
        
            for split_name, dist_dict, split_fn, split_kwargs in configs:
                for dataset_name, X_data in [('features', X_filtered), ('embeddings', embeddings_filtered)]:
                    for scenario in ['all', 'pca', 'relieff']:
                        splits = perform_multiple_splits(
                            split_fn, X_data, y_filtered, participant_ids_filtered,
                            n_iterations, split_name, **split_kwargs
                        )
                    
                        # Faz tuning independente em cada split
                        dist = evaluate_with_multiple_splits(
                            splits, prepare_all_scenarios, train_knn, calculate_metrics,
                            k_values_tuning, use_sklearn=USE_SKLEARN_KNN,
                            logger=logger, split_name=split_name, dataset_name=dataset_name
                        )
                    
                        # Calcula best_k usando moda dos 10 splits
                        from scipy import stats as scipy_stats
                        best_k_mode = int(scipy_stats.mode(dist[scenario]['best_ks'], keepdims=False)[0])
                    
                        dist_dict[dataset_name][scenario] = {
                            'f1_scores': dist[scenario]['f1_scores'],
                            'accuracies': dist[scenario]['accuracies'],
                            'best_k': best_k_mode,
                            'best_ks_per_split': dist[scenario]['best_ks'],
                            'confusion_matrices': dist[scenario]['confusion_matrices'],
                            'recalls_per_class': dist[scenario]['recalls_per_class']
                        }
                    
                        pbar.update(1)
        
            pbar.close()
        
            # Guardar resultados
            save_results(distributions_within, distributions_between)
        
        # Tabelas resumo com médias e desvios padrão
        print_summary_table(distributions_within, distributions_between, 'features')
        print_summary_table(distributions_within, distributions_between, 'embeddings')
        
        # Gerar matrizes de confusão médias para features_all
        print("\nGerando matrizes de confusão médias para FEATURES_ALL...")
        cm_path_within = plot_average_confusion_matrix(distributions_within, 'within', 'features')
        cm_path_between = plot_average_confusion_matrix(distributions_between, 'between', 'features')
        print(f"✓ Matriz de confusão (within-subject): {cm_path_within}")
        print(f"✓ Matriz de confusão (between-subject): {cm_path_between}")
        
        # Criar visualizações dos testes de hipótese
        print(f"\nGráficos salvos:")
        results_within_hyp, plot_within = plot_hypothesis_tests(distributions_within, 'within')
        results_between_hyp, plot_between = plot_hypothesis_tests(distributions_between, 'between')
        print(f"✓ {plot_within}")
        print(f"✓ {plot_between}")
        
        # Encontrar melhor modelo
        print("\n─── MELHORES MODELOS (Média de 10 iterações) ───")
        
        best_w, score_w, k_w, acc_w, _, comp_w = find_best_model(distributions_within)
        print(f"\nWithin-Subject:")
        print(f"  Melhor modelo: {best_w} (k={k_w})")
        print(f"  F1-Score médio: {score_w:.4f}")
        print(f"  Accuracy média: {acc_w:.4f}")
        sig_count = sum(1 for c in comp_w.values() if c['significant'])
        print(f"  Significativamente melhor que {sig_count}/{len(comp_w)} outros modelos")
        
        best_b, score_b, k_b, acc_b, _, comp_b = find_best_model(distributions_between)
        print(f"\nBetween-Subject:")
        print(f"  Melhor modelo: {best_b} (k={k_b})")
        print(f"  F1-Score médio: {score_b:.4f}")
        print(f"  Accuracy média: {acc_b:.4f}")
        sig_count = sum(1 for c in comp_b.values() if c['significant'])
        print(f"  Significativamente melhor que {sig_count}/{len(comp_b)} outros modelos")
        
        print("\nJustificação: Teste de Wilcoxon (paired, não-paramétrico)")
        print("  • Amostras emparelhadas (mesmo split em cenários diferentes)")
        print("  • Não assume normalidade da distribuição")
        print("  • Apropriado para comparar performance de modelos")
        
        execution_times['Exercício 5'] = time.time() - start_time
        print(f"\nTempo de execução: {format_time(execution_times['Exercício 5'])}")
        print("Exercício 5 concluído!")
        
        # EXERCÍCIO 6: Deployment e Avaliação de Accuracy
        print(f"\nEXERCÍCIO 6: DEPLOYMENT E AVALIAÇÃO DE ACCURACY")
        print("-" * 60)
        start_time = time.time()
        
        # Chama função para classificação com o modelo escolhido
        # Formato: '{split}_{dataset}_{scenario}'
        model_name = 'between_features_all'
        
        print(f"Modelo escolhido: {model_name}")
        print("Executando classificação de exemplo...\n")
        
        result = run_classification(
            model_name,
            distributions_within,
            distributions_between,
            X_filtered,
            embeddings_filtered,
            y_filtered,
            participant_ids_filtered
        )
        
        # Imprime resultados da classificação de exemplo
        print(f"\n✓ Exemplo de Classificação")
        print(f"  • k otimizado: {result['best_k']}")
        print(f"\n─── Seleção de Amostra de Teste ───")
        print(f"  • Participante selecionado: {result['participant']}")
        print(f"  • Atividade selecionada: {result['true_label']}")
        print(f"  • Ruído adicionado (desvio padrão médio): {result['noise_std']:.4f}")
        print(f"\n─── Dataset de Treino ───")
        print(f"  • Total de amostras (dataset completo): {result['n_train_samples']}")
        print(f"\n─── Balanceamento com SMOTE ───")
        print(f"  • Total de amostras balanceadas: {result['n_balanced_samples']}")
        print(f"\n─── Treinamento do Modelo ───")
        print(f"  • Modelo treinado com {result['n_balanced_samples']} amostras balanceadas")
        print(f"  • Array shape: {result['test_array_shape']}")
        print(f"  • Label verdadeira: Atividade {result['true_label']}")
        print(f"\n─── Classificação ───")
        print(f"  • Label predita: Atividade {result['predicted_label']}")
        print(f"  • Resultado: {'✓ CORRETA' if result['predicted_label'] == result['true_label'] else '✗ INCORRETA'}")
        
        # Avaliação de Accuracy com múltiplas iterações
        print(f"\n{'─' * 60}")
        print(f"AVALIAÇÃO DE ACCURACY")
        print(f"{'─' * 60}")

        n_iterations = 200

        print(f"Executando {n_iterations} classificações aleatórias...\n")
        
        evaluation = evaluate_deployment_accuracy(
            model_name=model_name,
            distributions_within=distributions_within,
            distributions_between=distributions_between,
            X_features=X_filtered,
            X_embeddings=embeddings_filtered,
            y_labels=y_filtered,
            participant_ids=participant_ids_filtered,
            n_iterations=n_iterations
        )
        
        print(f"\n{'=' * 60}")
        print(f"RESULTADOS DA AVALIAÇÃO")
        print(f"{'=' * 60}")
        print(f"Total de classificações: {evaluation['n_total']}")
        print(f"Classificações corretas: {evaluation['n_correct']}")
        print(f"Classificações incorretas: {evaluation['n_total'] - evaluation['n_correct']}")
        print(f"Accuracy: {evaluation['accuracy'] * 100:.2f}%")
        
        execution_times['Exercício 6'] = time.time() - start_time
        print(f"\nTempo de execução: {format_time(execution_times['Exercício 6'])}")
        print("Exercício 6 concluído!")
        
        # Resumo de tempos de execução
        print(f"\n{'=' * 60}")
        print(f"RESUMO DE TEMPOS DE EXECUÇÃO")
        print(f"{'=' * 60}")
        
        # Separa exercícios por meta
        meta1_exercises = [
            'Exercício 2', 'Exercício 3.1', 'Exercício 3.2', 'Exercício 3.3',
            'Exercício 3.4', 'Exercício 3.5', 'Exercícios 3.6 e 3.7',
            'Exercício 3.7.1', 'Exercício 4.1', 'Exercício 4.2',
            'Exercícios 4.3 e 4.4', 'Exercícios 4.5 e 4.6',
            'Exercício 4.6.1', 'Exercício 4.6.2'
        ]
        
        print("\nMETA 1 - ENGENHARIA DE CARACTERÍSTICAS")
        print("-" * 60)
        meta1_time = 0
        for exercise in meta1_exercises:
            if exercise in execution_times:
                exec_time = execution_times[exercise]
                print(f"{exercise:35s}: {format_time(exec_time)}")
                meta1_time += exec_time
        print(f"{'-' * 60}")
        print(f"{'Subtotal Meta 1':35s}: {format_time(meta1_time)}")
        print(f"{'-' * 60}")
        
        print("\nMETA 2 - TRANSFER LEARNING E DATA AUGMENTATION")
        print("-" * 60)
        meta2_time = 0
        for exercise, exec_time in execution_times.items():
            if exercise not in meta1_exercises:
                print(f"{exercise:35s}: {format_time(exec_time)}")
                meta2_time += exec_time
        print(f"{'-' * 60}")
        print(f"{'Subtotal Meta 2':35s}: {format_time(meta2_time)}")
        print(f"{'-' * 60}")
        
        total_time = meta1_time + meta2_time
        print(f"\n{'=' * 60}")
        print(f"{'TEMPO TOTAL':35s}: {format_time(total_time)}")
        print(f"{'=' * 60}")
        
        print(f"\nPROJETO CONCLUÍDO!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Erro: {e}")

if __name__ == "__main__":
    main()
