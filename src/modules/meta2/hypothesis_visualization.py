"""
Hypothesis Testing Visualization - Exercício 5.3
Visualização de testes estatísticos entre modelos.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


def plot_hypothesis_tests(distributions_dict, split_name, output_dir='plots/meta2/exercicio_5.3_hypothesis'):
    """
    Cria grid 5x3 de comparações par a par entre modelos usando KDE plots.
    
    Args:
        distributions_dict: Dict com {dataset: {scenario: {'f1_scores': [...], 'best_k': k}}}
        split_name: 'within' ou 'between'
        output_dir: Diretório para salvar gráficos
    
    Returns:
        results: Dict com resultados dos testes
    """
    from scipy.stats import gaussian_kde
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurações
    datasets = ['features', 'embeddings']
    scenarios = ['all', 'pca', 'relieff']
    
    # Preparar todos os pares de comparação (15 total)
    comparisons = []
    for ds1_idx, ds1 in enumerate(datasets):
        for sc1_idx, sc1 in enumerate(scenarios):
            for ds2_idx, ds2 in enumerate(datasets):
                for sc2_idx, sc2 in enumerate(scenarios):
                    # Evita comparar consigo mesmo
                    if (ds1_idx, sc1_idx) >= (ds2_idx, sc2_idx):
                        continue
                    comparisons.append((f"{ds1}_{sc1}", f"{ds2}_{sc2}"))
    
    # Criar figura 3x5
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    fig.suptitle(f'{split_name.upper()}-Subject Model Comparisons (All Pairs)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    results = {}
    
    for idx, (model1, model2) in enumerate(comparisons):
        row = idx // 5
        col = idx % 5
        ax = axes[row, col]
        
        # Extrair dados
        ds1, sc1 = model1.split('_')
        ds2, sc2 = model2.split('_')
        
        scores1 = np.array(distributions_dict[ds1][sc1]['f1_scores'])
        scores2 = np.array(distributions_dict[ds2][sc2]['f1_scores'])
        
        # Teste de Wilcoxon
        statistic, p_value = stats.wilcoxon(scores1, scores2)
        is_significant = p_value < 0.05
        
        # KDE para ambas as distribuições
        kde1 = gaussian_kde(scores1)
        kde2 = gaussian_kde(scores2)
        
        # Criar range para plot
        x_min = min(scores1.min(), scores2.min()) - 0.02
        x_max = max(scores1.max(), scores2.max()) + 0.02
        x_range = np.linspace(x_min, x_max, 200)
        
        # Plot KDE curves
        density1 = kde1(x_range)
        density2 = kde2(x_range)
        
        ax.fill_between(x_range, density1, alpha=0.4, color="#5DA5DA", label=f'{ds1.capitalize()}-{sc1.upper()}')
        ax.fill_between(x_range, density2, alpha=0.4, color="#FAA43A", label=f'{ds2.capitalize()}-{sc2.upper()}')
        
        # Linhas das curvas
        ax.plot(x_range, density1, color="#2E6F9E", linewidth=2, alpha=0.9)
        ax.plot(x_range, density2, color="#E67E22", linewidth=2, alpha=0.9)
        
        # Configurações dos eixos
        ax.set_xlabel('F1_score', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Anotação do p-value
        ax.text(0.98, 0.95, f'p={p_value:.4f}', 
                transform=ax.transAxes, fontsize=11, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Legenda
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        
        # Guardar resultado
        results[f'{model1}_vs_{model2}'] = {
            'p_value': p_value,
            'significant': is_significant,
            'mean1': np.mean(scores1),
            'mean2': np.mean(scores2)
        }
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'hypothesis_tests_{split_name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return results, output_path


def find_best_model(distributions_dict):
    """
    Identifica o melhor modelo com base em F1-score médio.
    
    Args:
        distributions_dict: Dict com distribuições
    
    Returns:
        best_model: Nome do melhor modelo
        best_score: F1-score médio
        best_k: Melhor k
        best_accuracy: Accuracy média
        models_scores: Dict com scores de todos os modelos
        comparison: Dict com comparações
    """
    best_model = None
    best_score = -np.inf
    best_k = None
    best_accuracy = None
    
    models_scores = {}
    
    for dataset in distributions_dict:
        for scenario in distributions_dict[dataset]:
            model_name = f"{dataset}_{scenario}"
            f1_scores = distributions_dict[dataset][scenario]['f1_scores']
            accuracies = distributions_dict[dataset][scenario].get('accuracies', [0]*len(f1_scores))
            k = distributions_dict[dataset][scenario]['best_k']
            mean_f1 = np.mean(f1_scores)
            mean_acc = np.mean(accuracies)
            
            models_scores[model_name] = {
                'mean_f1': mean_f1,
                'mean_accuracy': mean_acc,
                'std_f1': np.std(f1_scores),
                'std_accuracy': np.std(accuracies),
                'k': k
            }
            
            if mean_f1 > best_score:
                best_score = mean_f1
                best_accuracy = mean_acc
                best_model = model_name
                best_k = k
    
    # Testar significância do melhor vs outros
    best_scores = distributions_dict[best_model.split('_')[0]][best_model.split('_')[1]]['f1_scores']
    comparisons = {}
    
    for model_name in models_scores:
        if model_name == best_model:
            continue
        
        ds, sc = model_name.split('_')
        other_scores = distributions_dict[ds][sc]['f1_scores']
        _, p_value = stats.wilcoxon(best_scores, other_scores)
        
        comparisons[model_name] = {
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_f1': models_scores[model_name]['mean_f1'],
            'mean_accuracy': models_scores[model_name]['mean_accuracy']
        }
    
    return best_model, best_score, best_k, best_accuracy, models_scores, comparisons
