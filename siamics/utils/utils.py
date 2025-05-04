import umap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from io import BytesIO
from siamics.utils.futils import create_directories

def get_label_index(labels_list):
    map_dict={}
    types_unique = list(set(labels_list)) 
    map_dict = {lbl: idx for idx, lbl in enumerate(types_unique)}
    label_index = [map_dict[lbl] for lbl in labels_list]
    return label_index, types_unique

def plot_umap(
    data=None, 
    labels=None, 
    label_name_mapping=None, 
    application=None,
    n_classes=None,
    umap_embedding=None,
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
    if umap_embedding is None:
        # Step 1: Create and fit the UMAP model
        umap_model = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric=metric, **kwargs)
        umap_embedding = umap_model.fit_transform(data)

    # Step 2: Create the plot
    plt.figure(figsize=(8, 6))
    scatter_args = {'s': 10, 'alpha': 0.3}  #increased transparency of dots
    plt.title('UMAP Projection')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    if labels:
        # Map labels to class names if mapping is provided
        # unique_labels = sorted(set(labels))  # integers like 0, 1, 2, ..., 49

        if application == "source":
            # Hardcoded mapping
            label_name_mapping = {
                0: "GEO",
                1: "TCGA"
            }
            # 'GEO' -> 0
            name_to_index_mapping = {v: k for k, v in label_name_mapping.items()}
            try:
                labels = [name_to_index_mapping[label] for label in labels]
            except KeyError as e:
                raise ValueError(f"Unknown label {e} in source labels!")

        # Step 2: Prepare colors
        unique_labels = sorted(set(labels))

        # Generate color map for integer indices
        cmap = plt.cm.get_cmap('nipy_spectral', n_classes)
        index_to_color = {i: cmap(i) for i in range(n_classes)}
        colors = [index_to_color[label] for label in labels]

        if label_name_mapping is not None:
            try:
                display_names = {i: label_name_mapping[i] for i in unique_labels}
            except Exception as e:
                raise ValueError(f"Error mapping labels with `label_name_mapping`: {e}")
        else:
            display_names = {i: str(i) for i in unique_labels}


        plt.scatter(
            umap_embedding[:, 0], 
            umap_embedding[:, 1], 
            c=colors, 
            **scatter_args
        )
        
        handles = [Patch(color=index_to_color[label], label=display_names[label]) for label in unique_labels]
        if application:
            legend_title = "Centre" if application == "group_id" else application.replace("_", " ").title()

            plt.legend(
                handles=handles,
                title=legend_title,
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                fontsize=8,           # smaller text
                title_fontsize=9,     # smaller title
                markerscale=2.5,      # increase marker size relative to small dots
                handlelength=1.2,     # shorten line length
                handletextpad=0.4,    # space between marker and text
                borderpad=0.3,        # space inside legend box
                labelspacing=0.3,     # space between entries
                frameon=False         # remove box
            )

    else:
        plt.scatter(umap_embedding[:, 0], umap_embedding[:, 1], **scatter_args)
    
    # remove box, ticks, axes, axes labels
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel("")  
    ax.set_ylabel("")  
    for spine in ax.spines.values():
        spine.set_visible(False)

    # custom axes labels with arrows:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    arrowprops = dict(facecolor='black', arrowstyle='->', lw=0.8, shrinkA=0, shrinkB=0)

    # shorten arrows
    x_arrow_len = (xmax - xmin) * 0.05
    y_arrow_len = (ymax - ymin) * 0.05 
    arrowprops = dict(
        facecolor='black',
        arrowstyle='->',
        lw=0.8,
        shrinkA=0,
        shrinkB=0,
        mutation_scale=5  # smaller arrowhead
    )

    # Arrows 
    ax.annotate('', xy=(xmin + x_arrow_len, ymin), xytext=(xmin, ymin), arrowprops=arrowprops)  
    ax.annotate('', xy=(xmin, ymin + y_arrow_len), xytext=(xmin, ymin), arrowprops=arrowprops)  

    ax.text(xmin + x_arrow_len / 2, ymin - (ymax - ymin) * 0.02, "UMAP 1",
            va='top', ha='center', fontsize=10)

    ax.text(xmin - (xmax - xmin) * 0.03, ymin + y_arrow_len / 2, "UMAP 2",
            va='center', ha='right', fontsize=10, rotation=90)

            
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
    return image_buffer, umap_embedding

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
