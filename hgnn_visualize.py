import pandas as pd
import matplotlib.pyplot as plt

def plot_actual_vs_predicted(csv_filepath):
    try:
        df = pd.read_csv(csv_filepath, header=None, names=['actual', 'predicted'])
        
        plt.figure(figsize=(10, 6))
        plt.scatter(
            df['actual'], df['predicted'],
            alpha=0.8, s=60, color='dodgerblue',
            edgecolor='black', label='Actual vs. Predicted'
        )

        min_val = min(df['actual'].min(), df['predicted'].min())
        max_val = max(df['actual'].max(), df['predicted'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs. Predicted Values")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{csv_filepath}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{csv_filepath}' is empty.")
    except Exception as e:
        print(f"An error occurred: {e}")

plot_actual_vs_predicted('/Users/zaherlavi/Desktop/dev/thesis/GNN_EPSS_Thesis/test_logarithmic_actual_pred_output_2024.csv')