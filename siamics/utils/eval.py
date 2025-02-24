from sklearn import metrics 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from lifelines.utils import concordance_index
import numpy as np
from scipy.stats import pearsonr

class Classification: 

    def __init__(self, average='weighted', titles=None):
        self.average = average
        self.lbls = []
        self.preds = []
        self.titles = titles

    def gen_heatmap(self):
        # Create the heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.cm, annot=True, fmt="d", cmap="Blues", cbar=True, xticklabels=self.titles, yticklabels=self.titles)

        # Label axes
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')

        # Save plot as image
        plt.savefig("confusion_matrix.png")
        plt.close()

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

    def print(self):
        print(f"C-index: {self.c_index:.4f}")
        return self.c_index
    
class ImmuneDeconv:
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
        squared_error = (self.pred_prop - self.true_prop) ** 2
        sample_rmse = np.sqrt(np.mean(squared_error, axis=-1))  
        self.rmse = np.mean(sample_rmse)
        self.cell_type_pcc = [pearsonr(self.true_prop[:, i], self.pred_prop[:, i])[0] for i in range(self.true_prop.shape[1])]
        self.pcc = np.mean(self.cell_type_pcc)
        self.report = f"RMSE: {self.rmse}\nPCC: {self.pcc}"

    def cell_type_pcc_plot(self):
        num_cell_types = self.true_prop.shape[1]
        fig, axes = plt.subplots(nrows=1, ncols=num_cell_types, figsize=(5 * num_cell_types, 5))

        cell_types = ['B', 'CD4', 'CD8', 'NK', 'neutrophil', 'monocytic', 'fibroblasts', 'endothelial', 'others']

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
            ax.set_title(f"Cell Type {cell_types[i]} (PCC: {pcc:.2f})")
            ax.legend()

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
