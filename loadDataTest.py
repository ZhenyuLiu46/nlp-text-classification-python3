import random
import csv
import numpy as np

'''
#This works with non-excel generated .csv 
with open('Book2.csv', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)
'''

# do this to go around the excel-generated csv issue
with open('Book2.csv', newline='', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    your_list = list(reader)

print(your_list)
