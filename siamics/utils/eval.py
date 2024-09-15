from sklearn import metrics 
class Classification: 

    def __init__(self, data, gt_lbl, pred_lbl, average='weighted'):
        self.average = average
        self.gt_lbl = gt_lbl
        self.pred_lbl = pred_lbl
        
        self.update_metrics(data)

    def update_metrics(self, data):
        self.accuracy   = metrics.accuracy_score       (data[self.gt_lbl], data[self.pred_lbl])
        self.precision  = metrics.precision_score      (data[self.gt_lbl], data[self.pred_lbl], average=self.average)
        self.precision  = metrics.recall_score         (data[self.gt_lbl], data[self.pred_lbl], average=self.average)
        self.cm         = metrics.confusion_matrix     (data[self.gt_lbl], data[self.pred_lbl])
        self.report     = metrics.classification_report(data[self.gt_lbl], data[self.pred_lbl])
        
    def print(self):
        print(f"Classification Report: \n\n{self.report}")
        print(f"Confusion Matrix: \n{self.cm}")
