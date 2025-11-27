"""
Model Evaluation - Exercício 5
Avaliação de modelos k-NN com hyperparameter tuning e testes de hipótese.
"""

import numpy as np
from scipy import stats


def select_best_k(model_train_fn, X_train, y_train, X_val, y_val, 
                  k_values, metric_fn, use_sklearn=False, verbose=True):
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
        
    Returns:
        best_k: Melhor valor de k
        results: Lista com resultados para cada k
    """
    results = []
    best_k = None
    best_score = -np.inf
    
    if use_sklearn:
        from sklearn.neighbors import KNeighborsClassifier
    
    for k in k_values:
        if use_sklearn:
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(X_train, y_train)
        else:
            model = model_train_fn(X_train, y_train, k, verbose=False)
        y_pred = model.predict(X_val)
        metrics = metric_fn(y_val, y_pred, average='macro', verbose=False)
        score = metrics['f1_score']
        
        results.append({
            'k': k,
            'f1_score': score,
            'accuracy': metrics['accuracy']
        })
        
        if score > best_score:
            best_score = score
            best_k = k
    
    if verbose:
        print(f"Melhor k: {best_k} (F1={best_score:.4f})")
    
    return best_k, results


def train_and_evaluate(model_train_fn, X_train, y_train, X_val, y_val, 
                       X_test, y_test, best_k, metric_fn, use_sklearn=False):
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
        
    Returns:
        metrics: Dicionário com métricas no conjunto de teste
    """
    # Concatena train + val
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    # Treina com conjunto completo
    if use_sklearn:
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=best_k)
        model.fit(X_train_full, y_train_full)
    else:
        model = model_train_fn(X_train_full, y_train_full, best_k, verbose=False)
    
    # Avalia no teste
    y_pred = model.predict(X_test)
    metrics = metric_fn(y_test, y_pred, average='macro', verbose=False)
    
    return metrics


def evaluate_all_scenarios(scenarios_dict, model_train_fn, metric_fn, 
                           k_values, use_sklearn=False, verbose=True):
    """
    Avalia todos os cenários (all, pca, relieff).
    
    Args:
        scenarios_dict: Dicionário com cenários ('all', 'pca', 'relieff')
        model_train_fn: Função para treinar modelo
        metric_fn: Função para calcular métricas
        k_values: Lista de valores de k
        use_sklearn: Se True, usa KNeighborsClassifier do sklearn
        verbose: Se True, imprime progresso
        
    Returns:
        results: Dicionário com resultados por cenário
    """
    results = {}
    
    for scenario_name in ['all', 'pca', 'relieff']:
        scenario = scenarios_dict[scenario_name]
        
        # 1. Seleciona melhor k
        best_k, tuning_results = select_best_k(
            model_train_fn,
            scenario['X_train'],
            scenario['y_train'],
            scenario['X_val'],
            scenario['y_val'],
            k_values,
            metric_fn,
            use_sklearn=use_sklearn,
            verbose=False
        )
        
        # 2. Treina com train+val e avalia no teste
        test_metrics = train_and_evaluate(
            model_train_fn,
            scenario['X_train'],
            scenario['y_train'],
            scenario['X_val'],
            scenario['y_val'],
            scenario['X_test'],
            scenario['y_test'],
            best_k,
            metric_fn,
            use_sklearn=use_sklearn
        )
        
        results[scenario_name] = {
            'best_k': best_k,
            'tuning_results': tuning_results,
            'test_metrics': test_metrics
        }
    
    return results


def compare_confusion_matrices(results_dict, scenario_names, verbose=True):
    """
    Compara matrizes de confusão entre cenários.
    
    Args:
        results_dict: Dicionário com resultados
        scenario_names: Lista de nomes de cenários
        verbose: Se True, imprime análise
        
    Returns:
        analysis: Dicionário com análise
    """
    if not verbose:
        return {}
    
    print("\n─── Análise de Matrizes de Confusão ───")
    
    for name in scenario_names:
        cm = results_dict[name]['test_metrics']['confusion_matrix']
        classes = results_dict[name]['test_metrics']['classes']
        
        print(f"\n{name.upper()}:")
        
        # Identifica atividades mais difíceis (menor recall)
        recall_per_class = results_dict[name]['test_metrics']['recall_per_class']
        worst_idx = np.argmin(recall_per_class)
        worst_class = classes[worst_idx]
        worst_recall = recall_per_class[worst_idx]
        
        print(f"  Atividade mais difícil: {worst_class} (Recall={worst_recall:.3f})")
    
    return {}


def perform_multiple_splits(split_fn, X, y, participant_ids, n_iterations,
                            split_type, **split_kwargs):
    """
    Realiza múltiplas divisões train-val-test.
    
    Args:
        split_fn: Função de split (within ou between)
        X: Features
        y: Labels
        participant_ids: IDs dos participantes
        n_iterations: Número de iterações
        split_type: 'within' ou 'between'
        **split_kwargs: Argumentos adicionais para split_fn
        
    Returns:
        splits: Lista com n_iterations de splits
    """
    splits = []
    
    for i in range(n_iterations):
        split = split_fn(
            X=X,
            y=y,
            participant_ids=participant_ids,
            random_state=42 + i,
            **split_kwargs
        )
        splits.append(split)
    
    return splits


def evaluate_with_multiple_splits(splits_list, scenarios_prep_fn, 
                                  model_train_fn, metric_fn, best_k, use_sklearn=False):
    """
    Avalia modelo em múltiplas divisões (para distribuição de performance).
    
    Args:
        splits_list: Lista de splits
        scenarios_prep_fn: Função para preparar cenários
        model_train_fn: Função para treinar modelo
        metric_fn: Função para calcular métricas
        best_k: Valor de k a usar
        use_sklearn: Se True, usa KNeighborsClassifier do sklearn
        
    Returns:
        distributions: Dicionário com distribuições por cenário
    """
    distributions = {
        'all': {'f1_scores': [], 'accuracies': []},
        'pca': {'f1_scores': [], 'accuracies': []},
        'relieff': {'f1_scores': [], 'accuracies': []}
    }
    
    for split in splits_list:
        # Prepara cenários para este split
        scenarios = scenarios_prep_fn(
            split_data=split,
            variance_threshold=0.90,
            top_k_features=15,
            verbose=False
        )
        
        # Avalia cada cenário
        for scenario_name in ['all', 'pca', 'relieff']:
            scenario = scenarios[scenario_name]
            
            # Treina e avalia
            metrics = train_and_evaluate(
                model_train_fn,
                scenario['X_train'],
                scenario['y_train'],
                scenario['X_val'],
                scenario['y_val'],
                scenario['X_test'],
                scenario['y_test'],
                best_k,
                metric_fn,
                use_sklearn=use_sklearn
            )
            
            distributions[scenario_name]['f1_scores'].append(metrics['f1_score'])
            distributions[scenario_name]['accuracies'].append(metrics['accuracy'])
    
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
    
    if verbose:
        print(f"\n─── Testes de Hipótese (Wilcoxon, α={alpha}) ───")
    
    # Compara cada par de cenários
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
            
            if verbose:
                mean1 = np.mean(scores1)
                mean2 = np.mean(scores2)
                sig_marker = "***" if is_significant else "ns"
                print(f"  {name1} ({mean1:.4f}) vs {name2} ({mean2:.4f}): "
                      f"p={p_value:.4f} {sig_marker}")
    
    return results


def print_summary_table(results_within, results_between, dataset_name):
    """
    Imprime tabela resumo dos resultados.
    
    Args:
        results_within: Resultados do within-subject
        results_between: Resultados do between-subject
        dataset_name: Nome do dataset ('features' ou 'embeddings')
    """
    print(f"\n─── RESULTADOS: {dataset_name.upper()} ───")
    
    print(f"\n{'Cenário':<15} {'Split':<12} {'Best k':<8} {'F1-Score':<10} {'Accuracy':<10}")
    print("─" * 60)
    
    for scenario in ['all', 'pca', 'relieff']:
        # Within
        w = results_within[scenario]
        print(f"{scenario:<15} {'within':<12} {w['best_k']:<8} "
              f"{w['test_metrics']['f1_score']:<10.4f} "
              f"{w['test_metrics']['accuracy']:<10.4f}")
        
        # Between
        b = results_between[scenario]
        print(f"{'':<15} {'between':<12} {b['best_k']:<8} "
              f"{b['test_metrics']['f1_score']:<10.4f} "
              f"{b['test_metrics']['accuracy']:<10.4f}")
    
    print()
