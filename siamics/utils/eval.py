import os
from sklearn import metrics 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from scipy.stats import pearsonr, spearmanr
from .futils import create_directories
import jax.numpy as jnp
from matplotlib.lines import Line2D
from datetime import datetime
from sklearn.utils.class_weight import compute_sample_weight

def cccr(true_prop, pred_prop):
    mean_true = np.mean(true_prop)
    mean_pred = np.mean(pred_prop)
    var_true = np.var(true_prop)
    var_pred = np.var(pred_prop)
    covariance = np.mean((true_prop - mean_true) * (pred_prop - mean_pred))
    return (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred)**2)

class Classification: 

    def __init__(self, average='weighted', titles=None, ordinal=False, soft_tolerance=2):
        self.average = average
        self.lbls = []
        self.preds = []
        self.titles = titles
        self.ordinal=ordinal
        self.soft_tolerance=soft_tolerance

    def gen_heatmap(self, out_dir, figsize=(8, 6), filename=None, exclude_diagonal=False):

        if filename is None:
            # Get the current time as a string
            current_time = datetime.now().strftime("%H%M%S")

            # Append the timestamp to the filename
            filename = f"confusion_matrix_{current_time}.png"

        # Zero or mask diagonal if requested
        cm_to_plot = self.cm.copy()

        if exclude_diagonal:
            np.fill_diagonal(cm_to_plot, 0)  # or use np.nan to hide in heatmap

        # Create the heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(cm_to_plot, annot=True, fmt="d", cmap="Blues", cbar=True, xticklabels=self.titles, yticklabels=self.titles)

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
        self.weightedf1 = metrics.f1_score(self.lbls, self.preds, average='weighted')

        self.cm         = metrics.confusion_matrix     (self.lbls, self.preds)
        self.report     = metrics.classification_report(self.lbls, self.preds)
        if self.ordinal: 
            self.soft_accuracy, self.soft_class_accuracy = self._calc_soft_accuracy(tolerance = 2)
        
    def print(self, update=True):
        if update: 
            self.update_metrics()
        print(f"Classification Report: \n{self.titles}\n{self.report}")
        print(f"Weighted F1: \n{self.weightedf1}\n")
        
        print(f"Confusion Matrix: \n{self.titles}\n{self.cm}")
        if self.ordinal: 
            print(f"Soft Accuracy Tolerance: {self.soft_tolerance}")
            print(f"Overal Soft Accuracy: {self.soft_accuracy}")
            print(f"Per Class Soft Accuracy: {self.soft_class_accuracy}")

    def _calc_soft_accuracy(self, tolerance=2):
        """
        Computes accuracy where a prediction is considered correct if it's within 
        `tolerance` neighbors of the true class.
        
        Args:
            y_true (array-like): True class labels (as integers).
            y_pred (array-like): Predicted class labels (as integers).
            tolerance (int): Number of allowed neighbors (e.g., ±1).
        
        Returns:
            float: Tolerant accuracy score.
        """
        lbls = np.asarray(self.lbls)
        preds = np.asarray(self.preds)
        correct = np.abs(lbls - preds) <= tolerance
        # Calc overal accuracy. 
        overal_accuracy = np.sum(correct) / correct.shape[0]
        # Calc per class accuracy.  
        df = pd.DataFrame({'label': lbls, 'correct': correct})
        result = df.groupby('label')['correct'].mean()
        per_class_accuracy = result.to_dict()
        return overal_accuracy, per_class_accuracy

class ClassificationOnTheFly:
    def __init__(self, average='weighted', ordinal=False, soft_tolerance=2):
        self.average = average
        self.titles = []
        self.total_samples = 0
        self.correct_predictions = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        # Soft prediction for ordinal classification
        self.ordinal=ordinal
        self.soft_tolerance = soft_tolerance
        self.correct_soft_predictions = 0

    def get_titles(self):
        return self.titles
    
    def add_data(self, labels, preds): 
        self.total_samples += len(labels)
        self.correct_predictions += sum([1 for l, p in zip(labels, preds) if l == p])
        if self.ordinal: self.correct_soft_predictions += (np.abs(np.asarray(labels) - np.asarray(preds)) <= self.soft_tolerance).sum()
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
            self.soft_accuracy = self.correct_soft_predictions / self.total_samples
            self.precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
            self.recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
            self.cm = [[self.tn, self.fp], [self.fn, self.tp]]
            self.report = f"Accuracy: {self.accuracy}\nSoft Tolerance: {self.soft_tolerance}, Soft Accuracy: {self.soft_accuracy}\nPrecision: {self.precision}\nRecall: {self.recall}"
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

class Retrieval: 
    def __init__(self, titles=None):
        self.lbls = []
        self.preds = []
        self.titles = titles if titles else []

    def update(self, y_true, y_pred):
        self.lbls.extend(y_true)
        self.preds.extend(y_pred)

    def compute(self):
        assert len(self.lbls) == len(self.preds), "Mismatch in label and prediction length."
        results = {}

        # Overall metrics
        results['accuracy'] = metrics.accuracy_score(self.lbls, self.preds)
        results['f1_macro'] = metrics.f1_score(self.lbls, self.preds, average='macro')
        results['f1_weighted'] = metrics.f1_score(self.lbls, self.preds, average='weighted')
        sample_weights = compute_sample_weight(class_weight="balanced", y=self.lbls)
        results['weighted_accuracy'] = metrics.accuracy_score(self.lbls, self.preds, sample_weight=sample_weights)

        # Classwise metrics 
        report = metrics.classification_report(self.lbls, self.preds, output_dict=True, zero_division=0)
        lbls_arr = np.array(self.lbls)
        preds_arr = np.array(self.preds)
        unique_classes = np.unique(lbls_arr)

        results['classwise'] = {}
        for cls in unique_classes:
            cls_str = str(cls)
            cls_mask = lbls_arr == cls
            correct = np.sum(preds_arr[cls_mask] == cls)
            total = np.sum(cls_mask)
            accuracy_cls = correct / total if total > 0 else 0.0

            results['classwise'][cls_str] = {
                'precision': report[cls_str]['precision'],
                'recall': report[cls_str]['recall'],
                'f1': report[cls_str]['f1-score'],
                'support': report[cls_str]['support'],
                'accuracy': accuracy_cls 
            }

        return results

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
        
        # filter = (np.array(self.events) == 1)
        filtered_times = np.array(self.survival_time)
        filtered_scores = np.array(self.risk_score)

        self.c_index = concordance_index(filtered_times, -filtered_scores, event_observed=self.events)
        
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
        self.cell_type_rmse = [[]]
        self.cell_type_pcc = [[]]
        self.cell_type_scc = [[]]
        self.cell_type_ccc = [[]]
        self.sample_rmse = [[]]
        self.sample_pcc = [[]]
        self.sample_scc = [[]]
        self.sample_ccc = [[]]
    
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
        pred_prop = np.array(self.pred_prop)

        squared_error = (pred_prop - true_prop) ** 2

        # cell type level 
        n_celltypes = true_prop.shape[1]
        self.cell_type_rmse = np.sqrt(np.mean(squared_error, axis=0))  

        self.cell_type_pcc = []
        self.cell_type_scc = []
        self.cell_type_ccc = []

        for i in range(n_celltypes):
            t = true_prop[:, i]
            p = pred_prop[:, i]

            # detect constant inputs
            is_t_const = np.all(t == t[0])
            is_p_const = np.all(p == p[0])
            if is_t_const or is_p_const:
                which = []
                if is_t_const: which.append("true_prop")
                if is_p_const: which.append("pred_prop")
                print(f"[Warning] column {i} constant in: {', '.join(which)}")

            pcc_val = pearsonr(t, p)[0] if not (is_t_const or is_p_const) else np.nan
            scc_val = spearmanr(t, p)[0] if not (is_t_const or is_p_const) else np.nan
            ccc_val = cccr(t, p) if not (is_t_const or is_p_const) else np.nan

            self.cell_type_pcc.append(pcc_val)
            self.cell_type_scc.append(scc_val)
            self.cell_type_ccc.append(ccc_val)

        self.ct_rmse = np.mean(self.cell_type_rmse)
        self.ct_pcc = np.nanmean(self.cell_type_pcc)
        self.ct_scc = np.nanmean(self.cell_type_scc)
        self.ct_ccc = np.nanmean(self.cell_type_ccc)

        # sample level
        n_samples = true_prop.shape[0]
        self.sample_rmse = np.zeros(n_samples)
        self.sample_pcc  = np.zeros(n_samples)
        self.sample_scc  = np.zeros(n_samples)
        self.sample_ccc  = np.zeros(n_samples)

        for i in range(n_samples):
            t_i = true_prop[i, :]   
            p_i = pred_prop[i, :]

            self.sample_rmse[i] = np.sqrt(np.mean((p_i - t_i) ** 2))

            is_t_const = np.all(t_i == t_i[0])
            is_p_const = np.all(p_i == p_i[0])
            self.sample_pcc[i] = pearsonr(t_i, p_i)[0] if not (is_t_const or is_p_const) else np.nan
            self.sample_scc[i] = spearmanr(t_i, p_i)[0] if not (is_t_const or is_p_const) else np.nan
            self.sample_ccc[i] = cccr(t_i, p_i) if not (is_t_const or is_p_const) else np.nan

        self.mean_sample_rmse = np.nanmean(self.sample_rmse)
        self.mean_sample_pcc  = np.nanmean(self.sample_pcc)
        self.mean_sample_scc  = np.nanmean(self.sample_scc)
        self.mean_sample_ccc  = np.nanmean(self.sample_ccc)

        self.report = (
            f"Cell-type metrics:\n"
            f"  RMSE (avg over cell types): {self.ct_rmse:.4f}\n"
            f"  PCC (avg over cell types): {self.ct_pcc:.4f}\n"
            f"  SCC (avg over cell types): {self.ct_scc:.4f}\n"
            f"  CCC (avg over cell types): {self.ct_ccc:.4f}\n\n"
            f"Sample-level metrics (avg over samples):\n"
            f"  RMSE (avg over samples): {self.mean_sample_rmse:.4f}\n"
            f"  PCC (avg over samples): {self.mean_sample_pcc:.4f}\n"
            f"  SCC (avg over samples): {self.mean_sample_scc:.4f}\n"
            f"  CCC (avg over samples): {self.mean_sample_ccc:.4f}"
        )

    def cell_type_scatter_plot(self):
        num_cell_types = self.true_prop.shape[1]
        fig, axes = plt.subplots(nrows=1, ncols=num_cell_types, figsize=(5 * num_cell_types, 5))
        # cell_types = ['B cell', 'CD4+ T cell', 'CD8+ T cell', 'T cell (others)', 'NK cell', 'Granulocyte', 'Monocytic', 'Fibroblast', 'Endothelial cell', 'Others']
        # cell_types = ['B cell', 'CD4+ T cell', 'CD8+ T cell', 'NK cell', 'Neutrophil', 'Monocytic', 'Fibroblast', 'Endothelial cell', 'Others']
        cell_types = ['B cell', 'CD4+ T cell', 'CD8+ T cell', 'NK cell', 'Granulocyte', 'Monocytic', 'Fibroblast', 'Endothelial cell', 'Others']

        if num_cell_types == 1:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            x = self.true_prop[:, i]
            y = self.pred_prop[:, i]

            rmse_ct = self.cell_type_rmse[i]
            pcc_val = self.cell_type_pcc[i]
            scc_val = self.cell_type_scc[i]
            ccc_val = self.cell_type_ccc[i]

            invalid = (np.any(np.isnan(x)) or np.any(np.isnan(y)) or np.std(x) == 0 or np.std(y) == 0) #const pred or true

            if not invalid:
                ax.scatter(x, y, alpha=0.6)
                slope, intercept = np.polyfit(x, y, 1)
                ax.plot(x, slope * x + intercept, color="blue")
            else:
                ax.text(0.5, 0.5, "Invalid input", ha='center', va='center', transform=ax.transAxes)

            ax.set_xlabel("True Proportion")
            ax.set_ylabel("Predicted Proportion")
            ax.set_title(f"{cell_types[i]}")

            handles = [
                Line2D([], [], linestyle="none", label=f"PCC: {np.nan_to_num(pcc_val):.2f}"),
                Line2D([], [], linestyle="none", label=f"SCC: {np.nan_to_num(scc_val):.2f}"),
                Line2D([], [], linestyle="none", label=f"CCC: {np.nan_to_num(ccc_val):.2f}"),
                Line2D([], [], linestyle="none", label=f"RMSE: {rmse_ct:.2f}")
            ]
            legend = ax.legend(
                handles=handles,
                loc="upper right",
                frameon=True,             # draw the box
                framealpha=0.7,           # semi‐transparent
                edgecolor="black",
                fontsize="small",
                handlelength=1.0,    # shorten the little marker‐line on the left
                handletextpad=0.4,   # reduce space between the marker and the text
                labelspacing=1.0     # keep vertical spacing roughly at default
            )
            legend.get_frame().set_facecolor("white")

        plt.tight_layout()
        plt.savefig("cell_type_scatter.png")
        plt.close()

    def pcc_boxplot(self):
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=self.cell_type_pcc)
        plt.ylabel("PCC")
        plt.title("PCC Distribution")
        plt.savefig("pcc_boxplot.png")
        plt.close()

    def save_results(self, out_dir, model_name, cell_type_names: list[str]):
        np.savez(
            os.path.join(out_dir, "eval_res.npz"),
            model_name=model_name.split("_", 1)[1],
            true_prop=self.true_prop,
            pred_prop=self.pred_prop,
            cell_type_names=np.array(cell_type_names, dtype="U"),
        )

class GeneEssentiality:
    def __init__(self):
        self.true_prob = [[]]
        self.pred_prob = [[]]
        self.fingerprint = []

        self.mse = None
        self.pcc = None
        self.scc = None
        self.ccc = None

        self.perDep_scc = None
        self.perDep_pcc = None
        self.perDep_mse = None
        self.perDep_ccc = None
    
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
        self.pcc = pearsonr(self.true_prob.ravel(), self.pred_prob.ravel())[0]
        self.ccc = cccr(self.true_prob, self.pred_prob)
        
        # per-DepOI
        unique_genes = np.unique(self.fingerprint)
        gene_sccs = []
        gene_pccs = []
        gene_mses = []
        gene_cccs = []

        for gene in unique_genes:
            indices = [i for i, g in enumerate(self.fingerprint) if g == gene]
            indices_arr = jnp.array(indices)
            true_vector = true_prob[indices_arr, :].flatten()
            pred_vector = pred_prob[indices_arr, :].flatten()

            mse_val = float(np.mean((np.array(pred_vector) - np.array(true_vector)) ** 2))
            gene_mses.append(mse_val)

            if np.std(true_vector) != 0 and np.std(pred_vector) != 0:
                scc_val = spearmanr(np.array(true_vector), np.array(pred_vector))[0]
                pcc_val = pearsonr(np.array(true_vector), np.array(pred_vector))[0]
                ccc_val = cccr(np.array(true_vector), np.array(pred_vector))
            else:
                scc_val = np.nan
                pcc_val = np.nan
                ccc_val = np.nan
            
            gene_sccs.append(scc_val)
            gene_pccs.append(pcc_val)
            gene_cccs.append(ccc_val)

        self.perDep_scc = float(np.nanmean(gene_sccs))
        self.perDep_pcc = float(np.nanmean(gene_pccs))
        self.perDep_mse = float(np.nanmean(gene_mses))
        self.perDep_ccc = float(np.nanmean(gene_cccs))

        self.report = (
            f"SCC: {self.scc:.4f}\n"
            f"PCC: {self.pcc:.4f}\n"
            f"MSE: {self.mse:.4f}\n"
            f"CCC: {self.ccc:.4f}\n"
            f"per-DepOI SCC: {self.perDep_scc:.4f}\n"
            f"per-DepOI PCC: {self.perDep_pcc:.4f}\n"
            f"per-DepOI MSE: {self.perDep_mse:.4f}\n"
            f"per-DepOI CCC: {self.perDep_ccc:.4f}"
        )

    