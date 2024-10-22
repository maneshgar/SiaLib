from sklearn import metrics 
import pandas as pd
class Classification: 

    def __init__(self, average='weighted'):
        self.average = average
        self.lbls = []
        self.preds = []

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
