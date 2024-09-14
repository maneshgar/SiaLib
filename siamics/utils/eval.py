from sklearn import metrics 
class Classification: 

    def __init__(self, data, average='weighted'):
        self.average = average
        
        self.update_metrics(data)

    def update_metrics(self, data):
        self.accuracy = metrics.accuracy_score(data['gt'], data['pred'])
        self.precision = metrics.precision_score(data['gt'], data['pred'], average=self.average)
        self.precision = metrics.recall_score(data['gt'], data['pred'], average=self.average)
        self.cm = metrics.confusion_matrix(data['gt'], data['pred'])
        self.report = metrics.classification_report(data['gt'], data['pred'])
        
    def print(self):
        # print(f"Accuracy: {self.accuracy}")
        # print(f"Precision: {self.precision}")
        print(f"Classification Report: \n\n{self.report}")
        print(f"Confusion Matrix: \n{self.cm}")
