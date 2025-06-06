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

def class_colour(type="cancer_type"):
    values = {
        "cancer_type": ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD',
        'LUSC', 'MESO', 'OVARIAN', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM'],
        "subtype": ["Basal", "CMS2", "LuminalA", "CMS4", "LuminalB", "CMS3", "Classical", "HER2", "Normal", "CMS1", "Luminal"],
        "dataset": ["test", "TCGA", "GEO"],
        "centre": ["test_1", "1", "test_2",  "7", "test_3", "13", "test_4", "31", "test_5"]
    }

    return values.get(type, [])

def plot_umap(
    data=None, 
    labels=None, 
    # label_name_mapping=None, 
    label_type=None,
    # n_classes=None,
    umap_embedding=None,
    n_neighbors=15, 
    n_components=2, 
    metric='euclidean', 
    xlabel="X Label", 
    ylabel="Y Label", 
    save_path=None, 
    return_image=False, 
    transparency=0.8,
    figsize=(8, 6),
    s_size=4,
    plt_title="UMAP Projection",
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
        umap_model = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric=metric, random_state=42, **kwargs)
        umap_embedding = umap_model.fit_transform(data)

    # Step 2: Create the plot
    plt.figure(figsize=figsize)
    scatter_args = {'s': s_size, 'alpha': transparency, 'edgecolors': 'none'}  #increased transparency of dots
    plt.title(plt_title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    combined_palette = (
        list(plt.cm.get_cmap('tab20').colors) +
        list(plt.cm.get_cmap('tab20b').colors) +
        list(plt.cm.get_cmap('tab20c').colors) +
        list(plt.cm.get_cmap('Dark2').colors) +
        list(plt.cm.get_cmap('Pastel1').colors) +
        list(plt.cm.get_cmap('Pastel2').colors)
    )

    if labels is not None:
        # catch NaNs in labels
        orig_labels = []
        saw_nan = False
        for l in labels:
            if l is None or (isinstance(l, float) and np.isnan(l)):
                orig_labels.append("nan")
                saw_nan = True
            else:
                orig_labels.append(str(l))

        unique_labels = set(orig_labels)

        full_list = class_colour(label_type) or [] # get persis ind

        if label_type == "cluster": # numerical order for cluster
            numeric_labels = [lbl for lbl in unique_labels if lbl != "nan"]
            full_list = sorted(numeric_labels, key=lambda x: int(x))
        else:
            # append labels not found in defined labels
            for lbl in sorted(unique_labels):
                if lbl != "nan" and lbl not in full_list:
                    full_list.append(lbl)

        n_full = len(full_list)
        if n_full <= len(combined_palette):
            palette = combined_palette[:n_full]
        else:
            fallback = plt.cm.get_cmap('tab20', n_full) # use tab20 if number of classes > palette
            palette = [fallback(i) for i in range(n_full)]
        if saw_nan:
            palette.append((0.8, 0.8, 0.8))

        colors = [] # assign colours based on persis ind
        for lbl in orig_labels:
            if lbl == "nan":
                idx = n_full # gray
            else:
                idx = full_list.index(lbl)
            colors.append(palette[idx])

        plt.scatter(
            umap_embedding[:, 0], 
            umap_embedding[:, 1], 
            c=colors, 
            **scatter_args
        )
        
        # legend handles:
        handles = []
        legend_labels = []

        for cls in full_list:
            if cls in unique_labels:
                idx = full_list.index(cls)
                handles.append(
                    plt.Line2D([0], [0], marker='o', linestyle='', color='w', markerfacecolor=palette[idx], markersize=6)
                )
                legend_labels.append(str(cls))

        # add nan to legend if existed
        if saw_nan:
            handles.append(
                plt.Line2D([0], [0], marker='o', linestyle='', color='w', markerfacecolor=palette[n_full], markersize=6)
            )
            legend_labels.append("nan")

        # 4 col legend if more than 8 classes
        n_items = len(legend_labels)
        cols = 4 if n_items > 8 else 1
        plt.legend(
            handles,
            legend_labels,
            title=(label_type.replace('_', ' ') if label_type else ''),
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            ncol=cols,
            fontsize=8,           # smaller text
            title_fontsize=9,     # smaller title
            markerscale=1.5,      # increase marker size relative to small dots
            handlelength=1.0,     # shorten line length
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

def plot_hist(data, save_path=None):
    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.hist(data, bins=66, range=(0, 65), alpha=0.75, color='b', edgecolor='black')
    plt.xlabel("Bin Index")
    plt.ylabel("Frequency")
    plt.title("Histogram of Binned Token Frequencies")
    plt.xticks(np.arange(0, 65, step=5))  # Adjust ticks for readability
    if not save_path:
        save_path = "histogram.png"
    create_directories(save_path)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Histogram saved to {save_path}")
    return save_path

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
