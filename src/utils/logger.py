"""Logger para registar iterações do modelo."""

import os
from datetime import datetime


class ModelLogger:
    def __init__(self, output_dir='logs'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.tuning_file = os.path.join(output_dir, f'tuning_{timestamp}.log')
        self.final_file = os.path.join(output_dir, f'final_{timestamp}.log')
        
        # Headers
        with open(self.tuning_file, 'w') as f:
            f.write('split,dataset,scenario,iteration,k,f1_score,accuracy\n')
        with open(self.final_file, 'w') as f:
            f.write('split,dataset,scenario,iteration,best_k,f1_score,accuracy\n')
    
    def log_tuning(self, split, dataset, scenario, iteration, k, f1, acc):
        """Log cada teste de k no tuning."""
        with open(self.tuning_file, 'a') as f:
            f.write(f'{split},{dataset},{scenario},{iteration},{k},{f1:.4f},{acc:.4f}\n')
    
    def log_final(self, split, dataset, scenario, iteration, best_k, f1, acc):
        """Log avaliação final com melhor k."""
        with open(self.final_file, 'a') as f:
            f.write(f'{split},{dataset},{scenario},{iteration},{best_k},{f1:.4f},{acc:.4f}\n')
