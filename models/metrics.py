import matplotlib.pyplot as plt
import numpy as np
import torch

'''
Import metric libraries (torchmetrics, etc.) as necessary, create classes, etc.
'''

def metric_report(y_true: list[float], y_pred: list[float], *args, **kwargs):
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        print(y_true)
        print(y_pred)
    
    #create_plot(y_true, y_pred)

    return {'Metric': dummy_func(y_true, y_pred)}

def dummy_func(y_true, y_pred, *args, **kwargs):
    return NotImplementedError('Method Not Implemented')


def create_plot(y_true, y_pred, save_path="plot.png"):
    """
    Create a plot of predictions vs ground truth and save it to a file.

    Parameters:
    - y_true: Ground truth values.
    - y_pred: Predicted values.
    - save_path: File path to save the plot. Default is "plot.png".

    Returns:
    - None
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, c='blue', alpha=0.7, label='Predictions')
    
    # Add labels and title
    plt.title('Regression Plot')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')

    # Add a diagonal line for reference (perfect predictions)
    max_val = max(np.max(y_true), np.max(y_pred))
    min_val = min(np.min(y_true), np.min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='red', linewidth=2, label='Ideal')

    plt.grid(True)
    
    # Save the plot to the specified file path
    plt.savefig(save_path)
    plt.close()
