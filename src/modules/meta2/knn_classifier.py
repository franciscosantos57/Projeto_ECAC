"""
k-Nearest Neighbors Classifier - Exercício 4.1
Implementação própria do algoritmo k-NN para classificação.
"""

import numpy as np
from collections import Counter


class KNNClassifier:
    """
    Implementação simples de k-Nearest Neighbors (k-NN).
    """
    
    def __init__(self, k=3):
        """
        Inicializa o classificador k-NN.
        
        Args:
            k: Número de vizinhos mais próximos a considerar
        """
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X_train, y_train):
        """
        Treina o modelo (guarda os dados de treino).
        
        Args:
            X_train: Features de treino (n_samples, n_features)
            y_train: Labels de treino (n_samples,)
        """
        self.X_train = X_train
        self.y_train = y_train
        return self
    
    def predict(self, X_test):
        """
        Prediz as classes para os dados de teste.
        
        Args:
            X_test: Features de teste (n_samples, n_features)
            
        Returns:
            y_pred: Predições (n_samples,)
        """
        predictions = []
        
        for test_sample in X_test:
            # Calcula distâncias euclidianas para todos os pontos de treino
            distances = np.sqrt(np.sum((self.X_train - test_sample) ** 2, axis=1))
            
            # Encontra os k vizinhos mais próximos
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            # Voto por maioria
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        
        return np.array(predictions)
    
    def score(self, X_test, y_test):
        """
        Calcula a accuracy do modelo.
        
        Args:
            X_test: Features de teste
            y_test: Labels verdadeiras
            
        Returns:
            accuracy: Proporção de predições corretas
        """
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)


def train_knn(X_train, y_train, k=3, verbose=True):
    """
    Treina um classificador k-NN.
    
    Args:
        X_train: Features de treino
        y_train: Labels de treino
        k: Número de vizinhos
        verbose: Se True, imprime informações
        
    Returns:
        model: Modelo k-NN treinado
    """
    if verbose:
        print(f"Treinando k-NN com k={k}...")
        print(f"  Dados de treino: {X_train.shape[0]} amostras, {X_train.shape[1]} features")
    
    model = KNNClassifier(k=k)
    model.fit(X_train, y_train)
    
    if verbose:
        print(f"  ✓ Modelo treinado (memoriza {len(X_train)} exemplos)")
    
    return model


def predict_knn(model, X_test, verbose=True):
    """
    Realiza predições com um modelo k-NN.
    
    Args:
        model: Modelo k-NN treinado
        X_test: Features de teste
        verbose: Se True, imprime informações
        
    Returns:
        y_pred: Predições
    """
    if verbose:
        print(f"Predizendo {X_test.shape[0]} amostras...")
    
    y_pred = model.predict(X_test)
    
    if verbose:
        unique_classes = np.unique(y_pred)
        print(f"  ✓ Predições: {len(unique_classes)} classes distintas")
    
    return y_pred
