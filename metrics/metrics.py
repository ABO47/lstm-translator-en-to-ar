import numpy as np
import time
import psutil
import os
from tabulate import tabulate
import nltk
from nltk.translate.bleu_score import corpus_bleu
from data.data_utils import normalize_ar

nltk.download('punkt', quiet=True)


class MetricsTracker:
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.metrics = {}
    
    def track_training_metrics(self, history, train_duration):

        final_train_loss = history.history["loss"][-1]
        final_val_loss = history.history["val_loss"][-1]
        final_train_acc = history.history["accuracy"][-1]
        final_val_acc = history.history["val_accuracy"][-1]
        
        self.metrics['train_loss'] = final_train_loss
        self.metrics['val_loss'] = final_val_loss
        self.metrics['train_acc'] = final_train_acc
        self.metrics['val_acc'] = final_val_acc
        self.metrics['train_ppl'] = np.exp(final_train_loss)
        self.metrics['val_ppl'] = np.exp(final_val_loss)
        self.metrics['train_duration'] = train_duration
        self.metrics['train_ram'] = self.process.memory_info().rss / (1024 ** 2)
        self.metrics['train_cpu'] = psutil.cpu_percent(interval=1)
    
    def calculate_bleu_score(self, translator, en_texts, ar_texts, num_samples):
        references = [[normalize_ar(t).split()] for t in ar_texts[:num_samples]]
        candidates = [translator.translate(t).split() for t in en_texts[:num_samples]]
        bleu_score = corpus_bleu(references, candidates)
        self.metrics['bleu_score'] = bleu_score
        return bleu_score
    
    def track_inference_metrics(self, translator, en_texts, num_samples):
        start_time = time.time()
        for t in en_texts[:num_samples]:
            translator.translate(t)
        end_time = time.time()
        
        duration = end_time - start_time
        self.metrics['avg_inf_time'] = duration / num_samples
        self.metrics['inf_ram'] = self.process.memory_info().rss / (1024 ** 2)
        self.metrics['inf_cpu'] = psutil.cpu_percent(interval=1)
        self.metrics['throughput'] = num_samples / duration
    
    #display it as a table in the terminal
    def display_metrics(self):
        metrics_table = [
            ["Train Loss", f"{self.metrics.get('train_loss', 0):.4f}"],
            ["Validation Loss", f"{self.metrics.get('val_loss', 0):.4f}"],
            ["Train Accuracy", f"{self.metrics.get('train_acc', 0):.4f}"],
            ["Validation Accuracy", f"{self.metrics.get('val_acc', 0):.4f}"],
            ["Train Perplexity", f"{self.metrics.get('train_ppl', 0):.2f}"],
            ["Validation Perplexity", f"{self.metrics.get('val_ppl', 0):.2f}"],
            ["BLEU Score", f"{self.metrics.get('bleu_score', 0):.4f}"],
            ["Training Time (s)", f"{self.metrics.get('train_duration', 0):.2f}"],
            ["Training RAM (MB)", f"{self.metrics.get('train_ram', 0):.2f}"],
            ["Training CPU (%)", f"{self.metrics.get('train_cpu', 0)}"],
            ["Avg Inference Time / Sentence (s)", f"{self.metrics.get('avg_inf_time', 0):.4f}"],
            ["Inference RAM (MB)", f"{self.metrics.get('inf_ram', 0):.2f}"],
            ["Inference CPU (%)", f"{self.metrics.get('inf_cpu', 0)}"],
            ["Throughput (sent/sec)", f"{self.metrics.get('throughput', 0):.2f}"]
        ]
        
        print("\n=== MODEL METRICS ===")
        print(tabulate(metrics_table, headers=["Metric", "Value"], tablefmt="fancy_grid"))