import numpy as np
import math

class Score():
    def __init__(self, y_pred, y_true, size = 512, threshold = 0.5):
        self.TN = 0
        self.FN = 0
        self.FP = 0
        self.TP = 0
        self.y_pred = y_pred > threshold
        self.y_true = y_true
        self.threshold = threshold
        
        for i in range(0, size):
            for j in range(0, size):
                if self.y_pred[i,j] == 1:
                    if self.y_pred[i,j] == self.y_true[i][j]:
                        self.TP = self.TP + 1
                    else:
                        self.FP = self.FP + 1
                else:
                    if self.y_pred[i,j] == self.y_true[i][j]:
                        self.TN = self.TN + 1
                    else:
                        self.FN = self.FN + 1        
 
    def get_Se(self):
        return (self.TP)/(self.TP + self.FN)
    
    def get_Sp(self):
        return (self.TN)/(self.TN + self.FP)
    
    def get_Pr(self):
        return (self.TP)/(self.TP + self.FP)
    
    def F1(self):
        Pr = self.get_Pr()
        Se = self.get_Se()
        return (2*Pr*Se)/(Pr + Se)
    
    def G(self):
        Sp = self.get_Sp()
        Se = self.get_Se()
        return math.sqrt(Se*Sp)
    
    def IoU(self):
        Pr = self.get_Pr()
        Se = self.get_Se()
        return (Pr*Se) /(Pr + Se - Pr*Se)
    
    def DSC(self):
        return (2* self.TP)/(2* self.TP + self.FP + self.FN) 