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
