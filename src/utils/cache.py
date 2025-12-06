"""Cache para guardar/carregar resultados."""

import os
import pickle
from datetime import datetime


def save_results(distributions_within, distributions_between, output_dir='data/cache'):
    """Guarda resultados."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    data = {
        'distributions_within': distributions_within,
        'distributions_between': distributions_between,
        'timestamp': timestamp
    }
    
    filepath = os.path.join(output_dir, 'results.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✓ Resultados guardados: {filepath}")
    return filepath


def load_results(filepath='data/cache/results.pkl'):
    """Carrega resultados."""
    if not os.path.exists(filepath):
        return None
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✓ Resultados carregados: {filepath}")
    print(f"  Timestamp: {data['timestamp']}")
    return data['distributions_within'], data['distributions_between']


def cache_exists(filepath='data/cache/results.pkl'):
    """Verifica se existe cache."""
    return os.path.exists(filepath)
