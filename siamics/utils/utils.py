import umap
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from siamics.utils.futils import create_directories

def plot_umap(
    data, 
    labels=None, 
    n_neighbors=15, 
    n_components=2, 
    metric='euclidean', 
    xlabel="X Label", 
    ylabel="Y Label", 
    save_path=None, 
    return_image=False, 
    **kwargs
):
    """
    Creates a UMAP plot for the given data and optionally saves and returns the plot.

    Parameters:
        data (array-like): The high-dimensional input data.
        labels (array-like, optional): Labels for coloring the points.
        n_neighbors (int, optional): Number of neighbors for UMAP. Default is 15.
        n_components (int, optional): Number of UMAP dimensions. Default is 2.
        metric (str, optional): Distance metric for UMAP. Default is 'euclidean'.
        xlabel (str, optional): X-axis label. Default is "X Label".
        ylabel (str, optional): Y-axis label. Default is "Y Label".
        save_path (str, optional): File path to save the plot (e.g., "umap_plot.png").
        return_image (bool, optional): If True, returns the plot as a BytesIO object.
        **kwargs: Additional keyword arguments for UMAP or scatter plot.

    Returns:
        BytesIO or None: Returns BytesIO object if `return_image` is True, else None.
    """
    # Step 1: Create and fit the UMAP model
    umap_model = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric=metric, **kwargs)
    umap_embedding = umap_model.fit_transform(data)

    # Step 2: Create the plot
    plt.figure(figsize=(8, 6))
    scatter_args = {'s': 10, 'alpha': 0.7}  # Default scatter plot settings
    plt.title('UMAP Projection')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if labels:
        # Define a mapping from labels to colors
        unique_labels = list(set(labels))
        label_to_color = {label: color for label, color in zip(unique_labels, plt.cm.tab20.colors)}

        # Map the labels to colors
        colors = [label_to_color[label] for label in labels]

        scatter = plt.scatter(
            umap_embedding[:, 0], 
            umap_embedding[:, 1], 
            c=colors, 
            **scatter_args
        )
        plt.colorbar(scatter, label='Labels')

    else:
        plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], **scatter_args)
        
    # Step 3: Save the plot to disk if save_path is specified
    if save_path:
        create_directories(save_path)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    # Step 4: Save the plot to memory if return_image is True
    image_buffer = None
    if return_image:
        image_buffer = BytesIO()
        plt.savefig(image_buffer, format='png', bbox_inches='tight')
        image_buffer.seek(0)  # Move to the beginning of the BytesIO object

    # Show the plot
    plt.show()

    # Return the in-memory image if requested
    return image_buffer

def plot_hist(data):
    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=66, range=(0, 65), alpha=0.75, color='b', edgecolor='black')
    plt.xlabel("Bin Index")
    plt.ylabel("Frequency")
    plt.title("Histogram of Binned Token Frequencies")
    plt.xticks(np.arange(0, 65, step=5))  # Adjust ticks for readability
    plt.savefig("plt.png")

class RunningStats:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # Variance accumulator

    def update_batch(self, batch):
        """Updates running mean and std using a batch of data."""
        batch = np.array(batch)  # Ensure it's a NumPy array
        batch_size = len(batch)
        new_count = self.count + batch_size

        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, ddof=0, axis=0)  # Variance without Bessel correction

        # Compute delta between old mean and new batch mean
        delta = batch_mean - self.mean

        # Update running mean
        self.mean += (delta * batch_size) / new_count

        # Update variance accumulator M2
        self.M2 += batch_var * batch_size + (delta ** 2) * (self.count * batch_size) / new_count

        # Update count
        self.count = new_count

    def get_mean(self):
        return self.mean

    def get_std(self):
        return np.sqrt(self.M2 / self.count) if self.count > 1 else 0.0  # Population std (ddof=0)
