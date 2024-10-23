from sklearn import metrics 
import matplotlib.pyplot as plt
import seaborn as sns

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
        self.precision  = metrics.precision_score      (self.lbls, self.preds, average=self.average)
        self.recall     = metrics.recall_score         (self.lbls, self.preds, average=self.average)
        self.cm         = metrics.confusion_matrix     (self.lbls, self.preds)
        self.report     = metrics.classification_report(self.lbls, self.preds)
        
    def print(self, update=True):
        if update: 
            self.update_metrics()
        print(f"Classification Report: \n\n{self.report}")
        print(f"Confusion Matrix: \n{self.cm}")
