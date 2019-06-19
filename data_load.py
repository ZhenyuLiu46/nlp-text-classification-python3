import csv
import numpy as np
from config import *


class PurposeDataset():
    def __init__(self):
        self.file = PURPOSE_FILE
        self.all_data = self.LoadData()

    def LoadData(self):
        with open(PURPOSE_FILE, newline='', encoding='utf-8-sig') as f:  # For excel-generated csv issue
            reader = csv.reader(f)
            next(reader)  # This skips the first row of the CSV file.
            alert_list = list(reader)
        # alert_dic: dictionary use alertID(alert[0]) as the key, saves text and label
        alert_dic = {}
        # alert[0]: alert_id, alert[1]: text, alert[2]: label
        for alert in alert_list:
            if alert[0] in alert_dic:
                alert_dic[alert[0]][0] += "; " + alert[1]
            else:
                alert_dic[alert[0]] = [alert[1], alert[2]]
        # inputData as the list to save the text and label, [0] as the text description, [1] as the label
        inputData = [[]]
        for id, content in alert_dic.items():
            #print(id, ":", content)
            # change string label to int label(0,1)
            content[1] = int(content[1])
            inputData.append(content)

        return inputData


'''
#For non-excel generated .csv
with open('Book2.csv', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)
'''
