"""
Classification Metrics - Exercício 4.2
Função para calcular métricas de classificação.
"""

import numpy as np


def confusion_matrix(y_true, y_pred):
    """
    Calcula a matriz de confusão.
    
    Args:
        y_true: Labels verdadeiras
        y_pred: Labels preditas
        
    Returns:
        cm: Matriz de confusão (classes, classes)
        classes: Array com as classes ordenadas
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    
    # Cria matriz de confusão
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    
    return cm, classes


def calculate_metrics(y_true, y_pred, average='macro', verbose=False):
    """
    Calcula métricas de classificação.
    
    Args:
        y_true: Labels verdadeiras
        y_pred: Labels preditas
        average: Tipo de média para métricas multi-classe ('macro', 'weighted', 'micro')
        verbose: Se True, imprime as métricas
        
    Returns:
        metrics: Dicionário com as métricas calculadas
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Matriz de confusão
    cm, classes = confusion_matrix(y_true, y_pred)
    
    # Accuracy global
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    # Métricas por classe
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    support_per_class = []
    
    for i, cls in enumerate(classes):
        # True Positives, False Positives, False Negatives
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = np.sum(y_true == cls)
        
        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1-Score: 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
        support_per_class.append(support)
    
    # Agregação das métricas
    if average == 'macro':
        # Macro: média simples
        precision = np.mean(precision_per_class)
        recall = np.mean(recall_per_class)
        f1 = np.mean(f1_per_class)
    elif average == 'weighted':
        # Weighted: média ponderada pelo suporte
        total_support = np.sum(support_per_class)
        precision = np.sum([p * s for p, s in zip(precision_per_class, support_per_class)]) / total_support
        recall = np.sum([r * s for r, s in zip(recall_per_class, support_per_class)]) / total_support
        f1 = np.sum([f * s for f, s in zip(f1_per_class, support_per_class)]) / total_support
    elif average == 'micro':
        # Micro: calcula globalmente
        tp_total = np.sum([cm[i, i] for i in range(len(classes))])
        fp_total = np.sum(cm) - tp_total
        fn_total = fp_total
        
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        raise ValueError(f"Average inválido: {average}. Use 'macro', 'weighted' ou 'micro'.")
    
    metrics = {
        'confusion_matrix': cm,
        'classes': classes,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_per_class': np.array(precision_per_class),
        'recall_per_class': np.array(recall_per_class),
        'f1_per_class': np.array(f1_per_class),
        'support_per_class': np.array(support_per_class)
    }
    
    if verbose:
        print_metrics(metrics)
    
    return metrics


def print_metrics(metrics):
    """
    Imprime as métricas de classificação.
    
    Args:
        metrics: Dicionário com as métricas calculadas
    """
    print("\n─── MÉTRICAS DE CLASSIFICAÇÃO ───")
    
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    print("\n─── Matriz de Confusão ───")
    cm = metrics['confusion_matrix']
    classes = metrics['classes']
    
    # Cabeçalho
    print("       ", end="")
    for cls in classes:
        print(f"Pred {cls:2}", end="  ")
    print()
    
    # Linhas
    for i, cls in enumerate(classes):
        print(f"True {cls:2}", end=" ")
        for j in range(len(classes)):
            print(f"{cm[i, j]:7}", end="  ")
        print()
    
    print("\n─── Métricas por Classe ───")
    print(f"{'Classe':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("─" * 60)
    
    for i, cls in enumerate(classes):
        print(f"{cls:<10} {metrics['precision_per_class'][i]:<12.4f} "
              f"{metrics['recall_per_class'][i]:<12.4f} "
              f"{metrics['f1_per_class'][i]:<12.4f} "
              f"{metrics['support_per_class'][i]:<10.0f}")
    
    print()


def evaluate_classification(y_true, y_pred, average='macro', title=None):
    """
    Avalia um modelo de classificação e imprime as métricas.
    
    Args:
        y_true: Labels verdadeiras
        y_pred: Labels preditas
        average: Tipo de média ('macro', 'weighted', 'micro')
        title: Título opcional para a avaliação
        
    Returns:
        metrics: Dicionário com as métricas
    """
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    
    metrics = calculate_metrics(y_true, y_pred, average=average, verbose=True)
    
    return metrics
