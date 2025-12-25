import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def find_min_max_values(records):
    # Find the min and max values across records
    min_val = np.inf
    max_val = -np.inf
    for record in records:
        min_val = min(min_val, np.min(record))
        max_val = max(max_val, np.max(record))
    return min_val, max_val


def plot_combined_histograms(data_dict, title, xlabel, ylabel, num_bins=10, round_labels=False, save_path=None):
    # Determine bin range using the min/max across datasets
    all_data = np.concatenate(list(data_dict.values()))
    min_value, max_value = all_data.min(), all_data.max()

    # Slightly expand max to include edge cases
    max_value = max_value * 1.01  # Expand by 1% to include max

    # Define bin edges
    bins = np.linspace(min_value, max_value, num_bins + 1)
    if round_labels:
        bin_labels = [f'{int(bins[i])}-{int(bins[i + 1])}' for i in range(len(bins) - 1)]
    else:
        bin_labels = [f'{bins[i]:.2f}-{bins[i + 1]:.2f}' for i in range(len(bins) - 1)]

    plt.figure(figsize=(14, 10))

    width = (bins[1] - bins[0]) / (len(data_dict) + 1)  # Bar width
    colors = list(mcolors.TABLEAU_COLORS.values())  # Use TABLEAU_COLORS color scheme

    for i, (method, data) in enumerate(data_dict.items()):
        # Compute counts per bin for each dataset
        hist, _ = np.histogram(data, bins=bins)
        for j in range(len(hist)):
            if hist[j] > 0:  # Plot bars only for non-zero frequency
                plt.bar(bins[j] + i * width - width * len(data_dict) / 2, hist[j], width=width,
                        color=colors[i % len(colors)], label=method if j == 0 else "", alpha=0.7)

    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)

    # Set X-axis ticks to bin labels
    plt.xticks(bins[:-1] + width / 2, bin_labels, rotation=45, fontsize=20)

    # Set Y-axis range and ticks
    max_hist = max(max(np.histogram(data, bins=bins)[0]) for data in data_dict.values())
    plt.ylim(0, max_hist + 5)
    plt.yticks(fontsize=20)

    # Auto-select legend location
    plt.legend(loc='best', fontsize=20)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()  # Adjust layout
    if save_path:
        plt.savefig(save_path)
    plt.show()


def save_visualizations(utilization_records, n_matrix_records, save_dir='visualization_results'):
    num_iterations = len(utilization_records)
    # Ensure save directories exist
    utilization_dir = os.path.join(save_dir, 'utilization')
    n_matrix_dir = os.path.join(save_dir, 'n_matrix')
    os.makedirs(utilization_dir, exist_ok=True)
    os.makedirs(n_matrix_dir, exist_ok=True)

    # Evenly sample up to 4 iterations
    sample_indices = np.linspace(0, num_iterations - 1, min(4, num_iterations), dtype=int)

    utilization_samples = [utilization_records[i] for i in sample_indices]
    n_matrix_samples = [n_matrix_records[i] for i in sample_indices]

    # Use actual iteration numbers in labels
    utilization_data_dict = {f'Iteration {idx + 1}': sample.flatten() for idx, sample in
                             zip(sample_indices, utilization_samples)}
    n_matrix_data_dict = {f'Iteration {idx + 1}': sample.flatten() for idx, sample in
                          zip(sample_indices, n_matrix_samples)}

    # Save side-by-side histograms for utilization
    plot_combined_histograms(utilization_data_dict, 'Utilization Histogram', 'Utilization', 'Frequency', num_bins=10,
                             save_path=f'{utilization_dir}/Utilization_Histograms.png')

    # Save side-by-side histograms for n_matrix
    plot_combined_histograms(n_matrix_data_dict, 'n_matrix Histogram', 'Number of Fiber Links', 'Frequency',
                             num_bins=10,
                             save_path=f'{n_matrix_dir}/n_matrix_Histograms.png')


def plot_max_utilization_over_iterations(utilization_records, save_path1=None, save_path2=None):
    # Extract maximum bandwidth utilization per iteration
    max_utilizations = [np.max(utilization_records[i]) for i in range(len(utilization_records))]
    iterations = [f"{i // 2 + 1} {'TO' if i % 2 == 0 else 'RO'}" for i in range(len(utilization_records))]

    colors = list(mcolors.TABLEAU_COLORS.values())  # Use TABLEAU_COLORS color scheme

    # First plot: from the 1st iteration
    plt.figure(figsize=(14, 10))
    for i, (iteration, utilization) in enumerate(zip(iterations, max_utilizations)):
        color = colors[0] if i % 2 == 0 else colors[1]
        plt.plot(iteration, utilization, marker='o', color=color, markersize=8)
    plt.plot(iterations, max_utilizations, linestyle='-', color=colors[0], linewidth=2, markersize=8)
    plt.title('MLU Over Iterations (From 1st Iteration)', fontsize=24, weight='bold')
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('MLU', fontsize=20)
    plt.xticks(fontsize=16, rotation=45, ha='right')
    plt.yticks(fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(['After TO MLU', 'After RO MLU'], fontsize=18)
    plt.tight_layout()
    if save_path1:
        plt.savefig(save_path1, dpi=300)  # Save high-resolution image
    plt.show()

    # Second plot: from the 2nd iteration
    plt.figure(figsize=(14, 10))
    for i, (iteration, utilization) in enumerate(zip(iterations[2:], max_utilizations[2:])):
        color = colors[0] if (i + 2) % 2 == 0 else colors[1]
        plt.plot(iteration, utilization, marker='o', color=color, markersize=8)
    plt.plot(iterations[2:], max_utilizations[2:], linestyle='-', color=colors[0], linewidth=2, markersize=8)
    plt.title('MLU Over Iterations (From 2nd Iteration)', fontsize=24, weight='bold')
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('MLU', fontsize=20)
    plt.xticks(fontsize=16, rotation=45, ha='right')
    plt.yticks(fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(['After TO MLU', 'After RO MLU'], fontsize=18)
    plt.tight_layout()
    if save_path2:
        plt.savefig(save_path2, dpi=300)  # Save high-resolution image
    plt.show()
