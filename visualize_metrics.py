import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('pdf', 'svg')

def plot_metrics(csv_file, save_dir):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Set the plotting style
    sns.set(style="whitegrid")

    # Define the metrics to plot
    metrics = {
        "Training Losses": ["Train_Box_Loss", "Train_Cls_Loss", "Train_DFL_Loss"],
        "Validation Losses": ["Val_Box_Loss", "Val_Cls_Loss", "Val_DFL_Loss"],
        # "Precision and Recall": ["Precision", "Recall"],
        # "mAP Metrics": ["mAP50", "mAP50-95"]
    }

    # Create subplots
    num_plots = len(metrics)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 5 * num_plots))

    if num_plots == 1:
        axes = [axes]

    for ax, (title, cols) in zip(axes, metrics.items()):
        for col in cols:
            sns.lineplot(x="Epoch", y=col, data=df, marker="o", label=col, ax=ax)
        ax.set_title(f"{title} Over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()

    plt.tight_layout()

    # Save the figure
    plot_path = os.path.join(save_dir, "training_metrics.pdf")
    plt.savefig(plot_path)
    print(f"Saved training metrics plot at {plot_path}")

    # Optionally, display the plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize Training Metrics")
    parser.add_argument('--project-dir', type=str, required=True, help='Path to the project directory containing runs/train/')
    args = parser.parse_args()

    project_dir = args.project_dir
    csv_file = os.path.join(project_dir, "runs/train/training_metrics.csv")
    save_dir = os.path.join(project_dir, "runs/train")

    # Check if CSV exists
    if not os.path.isfile(csv_file):
        print(f"CSV file not found at {csv_file}. Ensure that training has been completed.")
        return

    # Plot metrics
    plot_metrics(csv_file, save_dir)

if __name__ == "__main__":
    main()