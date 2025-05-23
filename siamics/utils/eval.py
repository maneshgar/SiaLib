import os
from sklearn import metrics 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from scipy.stats import pearsonr, spearmanr
from .futils import create_directories
import jax.numpy as jnp

class Classification: 

    def __init__(self, average='weighted', titles=None):
        self.average = average
        self.lbls = []
        self.preds = []
        self.titles = titles

    def gen_heatmap(self, out_dir, filename="confusion_matrix.png"):
        # Create the heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.cm, annot=True, fmt="d", cmap="Blues", cbar=True, xticklabels=self.titles, yticklabels=self.titles)

        # Label axes
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')

        # Save plot as image
        create_directories(out_dir)
        plt_path = os.path.join(out_dir, filename)
        plt.savefig(plt_path)
        plt.close()
        return plt_path

    def get_groundTruth(self):
        return self.lbls
        
    def get_preds(self):
        return self.preds

    def get_titles(self):
        return self.titles
    
    def add_data(self, labels, preds): 
        self.lbls += labels
        self.preds += preds
        
    def update_metrics(self):
        self.accuracy   = metrics.accuracy_score       (self.lbls, self.preds)
        self.precision  = metrics.precision_score      (self.lbls, self.preds, average=self.average, zero_division=0)
        self.recall     = metrics.recall_score         (self.lbls, self.preds, average=self.average, zero_division=0)
        self.cm         = metrics.confusion_matrix     (self.lbls, self.preds)
        self.report     = metrics.classification_report(self.lbls, self.preds)
        
    def print(self, update=True):
        if update: 
            self.update_metrics()
        print(f"Classification Report: \n{self.titles}\n{self.report}")
        print(f"Confusion Matrix: \n{self.titles}\n{self.cm}")

class ClassificationOnTheFly:
    def __init__(self, average='weighted'):
        self.average = average
        self.titles = []
        self.total_samples = 0
        self.correct_predictions = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def get_titles(self):
        return self.titles
    
    def add_data(self, labels, preds): 
        self.total_samples += len(labels)
        self.correct_predictions += sum([1 for l, p in zip(labels, preds) if l == p])
        for l, p in zip(labels, preds):
            if l == p == 1:
                self.tp += 1
            elif l == 0 and p == 1:
                self.fp += 1
            elif l == 1 and p == 0:
                self.fn += 1
            elif l == p == 0:
                self.tn += 1

    def update_metrics(self):
        try:
            self.accuracy = self.correct_predictions / self.total_samples
            self.precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
            self.recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
            self.cm = [[self.tn, self.fp], [self.fn, self.tp]]
            self.report = f"Accuracy: {self.accuracy}\nPrecision: {self.precision}\nRecall: {self.recall}"
        except ZeroDivisionError:
            print("Warning: ZeroDivisionError occurred while updating metrics.")
            self.accuracy = 0
            self.precision = 0
            self.recall = 0
            self.cm = [[0, 0], [0, 0]]
            self.report = "Accuracy: 0\nPrecision: 0\nRecall: 0"
            
    def print(self, update=True):
        if update: 
            self.update_metrics()
        print(f"Classification Report: \n{self.titles}\n{self.report}")
        print(f"Confusion Matrix: \n{self.titles}\n{self.cm}")

class Survival:

    def __init__(self):
        self.c_index = -1
        self.survival_time = []
        self.risk_score = []
        self.events = []
        pass
    
    def add_data(self, survival_time, risk_score, events): 
        self.survival_time += survival_time
        self.risk_score += risk_score
        self.events += events
       
    def update_metrics(self):
        
        filter = (np.array(self.events) == 1)
        filtered_times = np.array(self.survival_time)[filter]
        filtered_scores = np.array(self.risk_score)[filter]

        self.c_index = concordance_index(filtered_times, -filtered_scores)
        
        return self.c_index
    
    def plot_KM(self, out_dir, filename="km_plot.png"):
        """
        Plots Kaplan-Meier survival curves for high-risk and low-risk groups.

        Parameters:
        - survival_time (array-like): Time until the event or censoring.
        - risk_score (array-like): Risk score to stratify high- and low-risk groups.
        - events (array-like): Binary event indicator (1 = event occurred, 0 = censored).

        Returns:
        - A Kaplan-Meier survival plot comparing high-risk and low-risk groups.
        """

        # Convert to numpy arrays
        survival_time = np.array(self.survival_time)
        risk_score = np.array(self.risk_score)
        events = np.array(self.events)

        # Define high- and low-risk groups using the median risk score
        median_risk = np.median(risk_score)
        high_risk_idx = risk_score >= median_risk
        low_risk_idx = risk_score < median_risk

        # Fit Kaplan-Meier survival curves
        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()

        plt.figure(figsize=(8, 6))

        # Fit and plot high-risk group
        kmf_high.fit(survival_time[high_risk_idx], events[high_risk_idx], label="High Risk")
        kmf_high.plot_survival_function(color="red", linestyle="--")

        # Fit and plot low-risk group
        kmf_low.fit(survival_time[low_risk_idx], events[low_risk_idx], label="Low Risk")
        kmf_low.plot_survival_function(color="blue", linestyle="-")

        # Perform log-rank test
        results = logrank_test(
            survival_time[high_risk_idx], survival_time[low_risk_idx],
            event_observed_A=events[high_risk_idx], event_observed_B=events[low_risk_idx]
        )
        
        p_value = results.p_value

        # Customize the plot
        plt.title(f"Kaplan-Meier Survival Curve (p = {p_value:.4f})")
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.legend()
        plt.grid(True)

        # Ensure save directory exists
        create_directories(out_dir)
        plot_path = os.path.join(out_dir, filename)

        # Save the plot
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()  # Close the plot to free memory

        return plot_path, p_value  # Return the file path and p-value
        
    def print(self, update=True):
        if update:
            self.update_metrics()
        print(f"C-index: {self.c_index:.4f}")
        return self.c_index
    
class TMEDeconv:
    def __init__(self):
        self.true_prop = [[]]
        self.pred_prop = [[]]
        self.cell_type_pcc = [[]]
    
    def add_data(self, true_p, pred_p): 
        true_p = np.array(true_p)  
        pred_p = np.array(pred_p)  

        if not hasattr(self, 'true_prop') or self.true_prop is None or isinstance(self.true_prop, list):
            self.true_prop = np.zeros((0, true_p.shape[1])) 

        if not hasattr(self, 'pred_prop') or self.pred_prop is None or isinstance(self.pred_prop, list):
            self.pred_prop = np.zeros((0, pred_p.shape[1])) 

        self.true_prop = np.vstack((self.true_prop, true_p))
        self.pred_prop = np.vstack((self.pred_prop, pred_p))

    def update_metrics(self):
        true_prop = np.array(self.true_prop)
        print(true_prop.shape)
        pred_prop = np.array(self.pred_prop)
        print(pred_prop.shape)

        squared_error = (pred_prop - true_prop) ** 2
        sample_rmse = np.sqrt(np.mean(squared_error, axis=-1))  
        self.rmse = np.mean(sample_rmse)
        self.cell_type_pcc = [pearsonr(true_prop[:, i], pred_prop[:, i])[0] for i in range(true_prop.shape[1])]
        self.pcc = np.mean(self.cell_type_pcc)
        self.report = f"RMSE: {self.rmse}\nPCC: {self.pcc}"

    def cell_type_pcc_plot(self):
        num_cell_types = self.true_prop.shape[1]
        fig, axes = plt.subplots(nrows=1, ncols=num_cell_types, figsize=(5 * num_cell_types, 5))

        cell_types = ['B cell', 'CD4+ T cell', 'CD8+ T cell', 'NK cell', 'Neutrophil', 'Monocyte', 'Fibroblast', 'Endothelial cell', 'Others']

        if num_cell_types == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            x = self.true_prop[:, i]
            y = self.pred_prop[:, i]

            pcc, _ = pearsonr(x, y)

            ax.scatter(x, y, alpha=0.6)

            slope, intercept = np.polyfit(x, y, 1)
            ax.plot(x, slope * x + intercept, color="blue")
            ax.set_xlabel("True Proportion")
            ax.set_ylabel("Predicted Proportion")
            ax.set_title(f"{cell_types[i]} (PCC: {pcc:.2f})")
            # ax.legend()

        plt.tight_layout()
        plt.savefig("cell_type_pcc.png")
        plt.close()

    def pcc_boxplot(self):
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=self.cell_type_pcc)
        plt.ylabel("PCC")
        plt.title("PCC Distribution")
        plt.savefig("pcc_boxplot.png")
        plt.close()

class GeneEssentiality:
    def __init__(self):
        self.true_prob = [[]]
        self.pred_prob = [[]]
        self.scc = [[]]
        self.fingerprint = []
    
    def add_data(self, true_p, pred_p, fingerprint): 
        true_p = np.array(true_p)  
        pred_p = np.array(pred_p)  

        if true_p.ndim == 1:
            true_p = true_p[:, None]
        if pred_p.ndim == 1:
            pred_p = pred_p[:, None]

        if not hasattr(self, 'true_prob') or self.true_prob is None or isinstance(self.true_prob, list):
            self.true_prob = np.zeros((0, true_p.shape[1])) 

        if not hasattr(self, 'pred_prob') or self.pred_prob is None or isinstance(self.pred_prob, list):
            self.pred_prob = np.zeros((0, pred_p.shape[1])) 

        self.true_prob = np.vstack((self.true_prob, true_p))
        self.pred_prob = np.vstack((self.pred_prob, pred_p))

        if not hasattr(self, 'fingerprint') or self.fingerprint is None:
            self.fingerprint = []
    
        if isinstance(fingerprint, list):
            self.fingerprint.extend(fingerprint)
        else:
            self.fingerprint.append(fingerprint)

    def update_metrics(self):
        true_prob = jnp.array(self.true_prob)
        pred_prob = jnp.array(self.pred_prob)
        squared_error = (pred_prob - true_prob) ** 2 
        self.mse = np.mean(squared_error)
        self.scc = spearmanr(self.true_prob, self.pred_prob)[0]
        
        # per-DepOI
        unique_genes = np.unique(self.fingerprint)
        gene_corrs = []
        for gene in unique_genes:
            indices = [i for i, g in enumerate(self.fingerprint) if g == gene]
            indices_arr = jnp.array(indices)
            true_vector = true_prob[indices_arr, :].flatten()
            pred_vector = pred_prob[indices_arr, :].flatten()
            if np.std(true_vector) != 0 and np.std(pred_vector) != 0:
                r, _ = spearmanr(true_vector, pred_vector)
            else:
                r = np.nan
            # r, _ = spearmanr(true_vector, pred_vector)
            gene_corrs.append(r)

        self.perDep_scc = np.nanmean(gene_corrs)

        self.report = f"MSE: {self.mse}\nSCC: {self.scc}\nper-DepOI: {self.perDep_scc}"

    