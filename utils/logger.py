"""
Logger utility for tracking training progress
"""

import torch
import numpy as np


class Logger:
    """Simple logger for tracking metrics across multiple runs."""

    def __init__(self, runs, metric='accuracy'):
        self.results = [[] for _ in range(runs)]
        self.metric = metric

    def add_result(self, run, result):
        """
        Add result for a specific run.

        Args:
            run (int): Run index
            result (tuple): (train_metric, val_metric, test_metric)
        """
        assert len(result) == 3
        assert 0 <= run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        """
        Print statistics for a specific run or all runs.

        Args:
            run (int, optional): Run index. If None, print stats for all runs.
        """
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train, valid, test))

            best_result = torch.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')

    def get_best_result(self):
        """
        Get the best result across all runs.

        Returns:
            dict: Best train, validation, and test metrics
        """
        result = 100 * torch.tensor(self.results)

        best_results = []
        for r in result:
            train = r[:, 0].max().item()
            valid = r[:, 1].max().item()
            test = r[r[:, 1].argmax(), 2].item()
            best_results.append((train, valid, test))

        best_result = torch.tensor(best_results)

        return {
            'train_mean': best_result[:, 0].mean().item(),
            'train_std': best_result[:, 0].std().item(),
            'valid_mean': best_result[:, 1].mean().item(),
            'valid_std': best_result[:, 1].std().item(),
            'test_mean': best_result[:, 2].mean().item(),
            'test_std': best_result[:, 2].std().item()
        }
