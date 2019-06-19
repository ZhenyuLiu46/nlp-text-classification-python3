import csv
import numpy as np
from os.path import realpath

'''
#For non-excel generated .csv
with open('Book2.csv', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)
'''

# For excel-generated csv issue
# filepath saves local file Book2.csv
filepath = realpath('../../Book2.csv')
print(filepath)
with open(filepath, newline='', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    # This skips the first row of the CSV file.
    next(reader)
    alert_list = list(reader)

# alert_dic: dictionary use alertID(alert[0]) as the key, saves text and label
alert_dic = {}
for alert in alert_list:
    if alert[0] in alert_dic:
        alert_dic[alert[0]][0] += "; " + alert[1]
    else:
        alert_dic[alert[0]] = [alert[1], alert[2]]
    # print(alert[0])

# inputData as the list to save the text and label
# input[0] as the text description
# inpit[1] as the label
inputData = [[]]
for id, content in alert_dic.items():
    #print(id, ":", content)
    content[1] = int(content[1])  # change string label to int label(0,1)
    inputData.append(content)


for data in inputData:
    if not data:
        continue
    else:
        print(data)
