import umap
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

    # Define a mapping from labels to colors
    unique_labels = list(set(labels))
    label_to_color = {label: color for label, color in zip(unique_labels, plt.cm.tab20.colors)}

    # Map the labels to colors
    colors = [label_to_color[label] for label in labels]

    # Step 2: Create the plot
    plt.figure(figsize=(8, 6))
    scatter_args = {'s': 10, 'alpha': 0.7}  # Default scatter plot settings
    if colors is not None:
        scatter = plt.scatter(
            umap_embedding[:, 0], 
            umap_embedding[:, 1], 
            c=colors, 
            **scatter_args
        )
        plt.colorbar(scatter, label='Labels')
    else:
        plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], **scatter_args)
        
    plt.title('UMAP Projection')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

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