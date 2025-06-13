import pandas as pd
import matplotlib.pyplot as plt

def plot_actual_vs_predicted(actual, predicted, title, file_prefix, epoch):
    # full version
    plt.figure(figsize=(10, 6))
    plt.scatter(
        actual, predicted,
        alpha=0.8, s=60, color='dodgerblue',
        edgecolor='black', label='Actual vs. Predicted'
    )


    plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Prediction')

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f'{title} - Full Range')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{file_prefix}_{epoch}_full.png', dpi=300)
    plt.show()


    # zoomed
    plt.figure(figsize=(10, 6))
    plt.scatter(
        actual, predicted,
        alpha=0.8, s=60, color='dodgerblue',
        edgecolor='black', label='Actual vs. Predicted'
    )


    plt.plot([0, 0.1], [0, 0.1], 'r--', label='Perfect Prediction')
    plt.xlim(0, 0.1)
    plt.ylim(0, 0.1)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(f'{title} - Zoomed (0 to 0.1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{file_prefix}_{epoch}_zoomed.png', dpi=300)
    plt.show()




def plot_actual_vs_predicted(actual, predicted, title, file_prefix, epoch):
    # Non-Zoomed
    plt.figure(figsize=(10, 6))
    plt.scatter(actual, predicted, color='blue', alpha=0.6, edgecolors='k', label='Actual vs. Predicted')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{title} - Full Range')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{file_prefix}_{epoch}_full.png', dpi=300)
    plt.show()

    # Zoomed
    plt.figure(figsize=(10, 6))
    plt.scatter(actual, predicted, color='blue', alpha=0.6, edgecolors='k', label='Actual vs. Predicted')
    plt.plot([0, 0.1], [0, 0.1], 'r--', label='Perfect Prediction')
    plt.xlim(0, 0.1)
    plt.ylim(0, 0.1)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{title} - Zoomed (0 to 0.1)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{file_prefix}_{epoch}_zoomed.png', dpi=300)
    plt.show()


plot_actual_vs_predicted('/Users/zaherlavi/Desktop/dev/thesis/GNN_EPSS_Thesis/test_logarithmic_actual_pred_output_2024.csv')