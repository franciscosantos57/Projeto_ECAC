"""
Model Evaluation - Exercício 5
Avaliação de modelos k-NN com hyperparameter tuning e testes de hipótese.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from .smote_balancer import balance_dataset_smote


def select_best_k(model_train_fn, X_train, y_train, X_val, y_val, 
                  k_values, metric_fn, use_sklearn=False, verbose=True, use_smote=True, logger=None, log_ctx=None):
    """
    Seleciona o melhor k usando dados de treino e validação.
    
    Args:
        model_train_fn: Função para treinar modelo (recebe X, y, k)
        X_train: Features de treino
        y_train: Labels de treino
        X_val: Features de validação
        y_val: Labels de validação
        k_values: Lista de valores de k a testar
        metric_fn: Função para calcular métrica (recebe y_true, y_pred)
        use_sklearn: Se True, usa KNeighborsClassifier do sklearn
        verbose: Se True, imprime progresso
        use_smote: Se True, balanceia treino com SMOTE
        logger: ModelLogger instance
        log_ctx: Dicionário com contexto de logging (split, dataset, scenario, iteration)
        
    Returns:
        best_k: Melhor valor de k
        results: Lista com resultados para cada k
    """
    # Balanceia com SMOTE se solicitado
    if use_smote:
        X_train, y_train = balance_dataset_smote(X_train, y_train, n_neighbors=5, verbose=False)
    
    results = []
    best_k = None
    best_score = -np.inf
    
    if use_sklearn:
        from sklearn.neighbors import KNeighborsClassifier
    
    for k in k_values:
        if use_sklearn:
            # sklearn com Manhattan + weighted (melhorias aplicadas)
            model = KNeighborsClassifier(n_neighbors=k, metric='manhattan', weights='distance', n_jobs=1)
            model.fit(X_train, y_train)
        else:
            model = model_train_fn(X_train, y_train, k, verbose=False)
        y_pred = model.predict(X_val)
        metrics = metric_fn(y_val, y_pred, average='macro', verbose=False)
        score = metrics['f1_score']
        
        results.append({'k': k, 'f1_score': score, 'accuracy': metrics['accuracy']})
        
        if logger and log_ctx:
            logger.log_tuning(log_ctx['split'], log_ctx['dataset'], log_ctx['scenario'], 
                            log_ctx['iteration'], k, score, metrics['accuracy'])
        
        if score > best_score:
            best_score = score
            best_k = k
    
    if verbose:
        print(f"Melhor k: {best_k} (F1={best_score:.4f})")
    
    return best_k, results


def train_and_evaluate(model_train_fn, X_train, y_train, X_val, y_val, 
                       X_test, y_test, best_k, metric_fn, use_sklearn=False, use_smote=True):
    """
    Treina modelo com train+val e avalia no teste.
    
    Args:
        model_train_fn: Função para treinar modelo
        X_train: Features de treino
        y_train: Labels de treino
        X_val: Features de validação
        y_val: Labels de validação
        X_test: Features de teste
        y_test: Labels de teste
        best_k: Valor de k selecionado
        metric_fn: Função para calcular métricas
        use_sklearn: Se True, usa KNeighborsClassifier do sklearn
        use_smote: Se True, balanceia treino com SMOTE
        
    Returns:
        metrics: Dicionário com métricas no conjunto de teste
    """
    # CORREÇÃO: Aplica SMOTE apenas no train, depois concatena com val
    # Isso evita data leakage (SMOTE não deve "ver" dados de validação)
    if use_smote:
        X_train_balanced, y_train_balanced = balance_dataset_smote(X_train, y_train, n_neighbors=5, verbose=False)
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Concatena train balanceado + val original
    X_train_full = np.vstack([X_train_balanced, X_val])
    y_train_full = np.concatenate([y_train_balanced, y_val])
    
    # Treina com conjunto completo
    if use_sklearn:
        from sklearn.neighbors import KNeighborsClassifier
        # sklearn com Manhattan + weighted (melhorias aplicadas)
        model = KNeighborsClassifier(n_neighbors=best_k, metric='manhattan', weights='distance', n_jobs=1)
        model.fit(X_train_full, y_train_full)
    else:
        model = model_train_fn(X_train_full, y_train_full, best_k, verbose=False)
    
    # Avalia no teste
    y_pred = model.predict(X_test)
    metrics = metric_fn(y_test, y_pred, average='macro', verbose=False)
    
    return metrics


def perform_multiple_splits(split_fn, X, y, participant_ids, n_iterations,
                            split_type, **split_kwargs):
    """Realiza múltiplas divisões train-val-test."""
    splits = []
    
    for i in range(n_iterations):
        split = split_fn(
            X=X, y=y, participant_ids=participant_ids,
            random_state=42 + i, **split_kwargs
        )
        splits.append(split)
    
    return splits


def evaluate_with_multiple_splits(splits_list, scenarios_prep_fn, 
                                  model_train_fn, metric_fn, k_values, use_sklearn=False, use_smote=True,
                                  logger=None, split_name=None, dataset_name=None):
    """Avalia modelo em múltiplas divisões com tuning de k independente."""
    distributions = {
        'all': {'f1_scores': [], 'accuracies': [], 'best_ks': [], 'confusion_matrices': [], 'recalls_per_class': []},
        'pca': {'f1_scores': [], 'accuracies': [], 'best_ks': [], 'confusion_matrices': [], 'recalls_per_class': []},
        'relieff': {'f1_scores': [], 'accuracies': [], 'best_ks': [], 'confusion_matrices': [], 'recalls_per_class': []}
    }
    
    for iteration, split in enumerate(splits_list, start=1):
        scenarios = scenarios_prep_fn(
            split_data=split, variance_threshold=0.90, top_k_features=15, verbose=False
        )
        
        for scenario_name in ['all', 'pca', 'relieff']:
            scenario = scenarios[scenario_name]
            
            log_ctx = {'split': split_name, 'dataset': dataset_name, 
                      'scenario': scenario_name, 'iteration': iteration} if logger else None
            
            # Faz tuning de k independentemente para este split
            best_k, _ = select_best_k(
                model_train_fn, scenario['X_train'], scenario['y_train'],
                scenario['X_val'], scenario['y_val'], k_values, metric_fn,
                use_sklearn=use_sklearn, verbose=False, use_smote=use_smote,
                logger=logger, log_ctx=log_ctx
            )
            
            # Treina e avalia com o melhor k deste split
            metrics = train_and_evaluate(
                model_train_fn, scenario['X_train'], scenario['y_train'],
                scenario['X_val'], scenario['y_val'], scenario['X_test'], scenario['y_test'],
                best_k, metric_fn, use_sklearn=use_sklearn, use_smote=use_smote
            )
            
            if logger and log_ctx:
                logger.log_final(split_name, dataset_name, scenario_name, iteration,
                               best_k, metrics['f1_score'], metrics['accuracy'])
            
            distributions[scenario_name]['f1_scores'].append(metrics['f1_score'])
            distributions[scenario_name]['accuracies'].append(metrics['accuracy'])
            distributions[scenario_name]['best_ks'].append(best_k)
            distributions[scenario_name]['confusion_matrices'].append(metrics['confusion_matrix'])
            distributions[scenario_name]['recalls_per_class'].append(metrics['recall_per_class'])
    
    return distributions


def hypothesis_testing(distributions_dict, alpha=0.05, verbose=True):
    """
    Testa hipóteses entre modelos usando teste de Wilcoxon.
    
    Args:
        distributions_dict: Dicionário com distribuições
        alpha: Nível de significância
        verbose: Se True, imprime resultados
        
    Returns:
        results: Dicionário com resultados dos testes
    """
    scenario_names = list(distributions_dict.keys())
    results = {}
    
    for i in range(len(scenario_names)):
        for j in range(i + 1, len(scenario_names)):
            name1 = scenario_names[i]
            name2 = scenario_names[j]
            
            scores1 = distributions_dict[name1]['f1_scores']
            scores2 = distributions_dict[name2]['f1_scores']
            
            # Teste de Wilcoxon (paired, não-paramétrico)
            statistic, p_value = stats.wilcoxon(scores1, scores2)
            
            is_significant = p_value < alpha
            
            results[f"{name1}_vs_{name2}"] = {
                'statistic': statistic,
                'p_value': p_value,
                'significant': is_significant
            }
    
    return results


def print_summary_table(distributions_within, distributions_between, dataset_name):
    """
    Imprime tabela resumo dos resultados de múltiplas iterações (Exercício 5.3).
    Mostra média, desvio padrão, melhor k (moda) e atividade mais difícil das 10 iterações.
    
    Args:
        distributions_within: Dict com distribuições within-subject
        distributions_between: Dict com distribuições between-subject
        dataset_name: Nome do dataset ('features' ou 'embeddings')
    """
    from scipy import stats as scipy_stats
    
    print(f"\n─── RESULTADOS: {dataset_name.upper()} ───")
    
    print(f"\n{'Cenário':<15} {'Split':<12} {'Best k':<8} {'F1-Score (±σ)':<20} {'Accuracy (±σ)':<20} {'Ativ. Difícil (Recall)'}")
    print("─" * 105)
    
    for scenario in ['all', 'pca', 'relieff']:
        # Within-Subject
        w = distributions_within[dataset_name][scenario]
        f1_mean = np.mean(w['f1_scores'])
        f1_std = np.std(w['f1_scores'])
        acc_mean = np.mean(w['accuracies'])
        acc_std = np.std(w['accuracies'])
        best_k_mode = int(scipy_stats.mode(w['best_ks_per_split'], keepdims=False)[0])
        
        # Calcular recall médio por classe e encontrar pior atividade
        recalls_array = np.array(w['recalls_per_class'])  # shape: (n_iterations, n_classes)
        mean_recalls = np.mean(recalls_array, axis=0)
        worst_class_idx = np.argmin(mean_recalls)
        worst_recall = mean_recalls[worst_class_idx]
        
        print(f"{scenario:<15} {'within':<12} {best_k_mode:<8} "
              f"{f1_mean:.4f} (±{f1_std:.4f})     "
              f"{acc_mean:.4f} (±{acc_std:.4f})     "
              f"Ativ. {worst_class_idx + 1} ({worst_recall:.3f})")
        
        # Between-Subject
        b = distributions_between[dataset_name][scenario]
        f1_mean = np.mean(b['f1_scores'])
        f1_std = np.std(b['f1_scores'])
        acc_mean = np.mean(b['accuracies'])
        acc_std = np.std(b['accuracies'])
        best_k_mode = int(scipy_stats.mode(b['best_ks_per_split'], keepdims=False)[0])
        
        # Calcular recall médio por classe e encontrar pior atividade
        recalls_array = np.array(b['recalls_per_class'])  # shape: (n_iterations, n_classes)
        mean_recalls = np.mean(recalls_array, axis=0)
        worst_class_idx = np.argmin(mean_recalls)
        worst_recall = mean_recalls[worst_class_idx]
        
        print(f"{'':<15} {'between':<12} {best_k_mode:<8} "
              f"{f1_mean:.4f} (±{f1_std:.4f})     "
              f"{acc_mean:.4f} (±{acc_std:.4f})     "
              f"Ativ. {worst_class_idx + 1} ({worst_recall:.3f})")
    
    print()


def plot_average_confusion_matrix(distributions_dict, split_type, dataset_name, 
                                  output_dir='plots/meta2/exercicio_5.2_confusion_matrices'):
    """
    Plota matriz de confusão média para o cenário 'all' das 10 iterações.
    
    Args:
        distributions_dict: Dict com distribuições (from evaluate_with_multiple_splits)
        split_type: 'within' ou 'between'
        dataset_name: 'features' ou 'embeddings'
        output_dir: Diretório para salvar o gráfico
        
    Returns:
        str: Caminho do arquivo salvo
    """
    from scipy import stats as scipy_stats
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Usa apenas cenário 'all'
    scenario_data = distributions_dict[dataset_name]['all']
    
    # Calcular matriz de confusão média
    confusion_matrices = np.array(scenario_data['confusion_matrices'])  # shape: (n_iterations, n_classes, n_classes)
    cm_mean = np.mean(confusion_matrices, axis=0)
    
    # Calcular métricas médias
    best_k_mode = int(scipy_stats.mode(scenario_data['best_ks_per_split'], keepdims=False)[0])
    f1_mean = np.mean(scenario_data['f1_scores'])
    acc_mean = np.mean(scenario_data['accuracies'])
    
    # Classes (atividades numeradas de 1 a n_classes)
    n_classes = cm_mean.shape[0]
    classes = list(range(1, n_classes + 1))
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plotar matriz de confusão média
    sns.heatmap(cm_mean, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Média de amostras (10 iterações)'},
                ax=ax, square=True, linewidths=0.5, linecolor='gray')
    
    # Configurações
    ax.set_xlabel('Classe Predita', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classe Real', fontsize=12, fontweight='bold')
    
    title = f'Matriz de Confusão Média - {dataset_name.upper()}_ALL ({split_type.upper()}-Subject)\n'
    title += f'k={best_k_mode} | Accuracy={acc_mean:.4f} | F1-Score={f1_mean:.4f}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Salvar
    filename = f'confusion_matrix_{dataset_name}_all_{split_type}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

