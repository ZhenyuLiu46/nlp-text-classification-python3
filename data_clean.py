import data_load

inputData = data_load.PurposeDataset()

# Test function and print
#inputData = DataLoad.LoadData()
for data in inputData.all_data:
    if not data:
        continue
    else:
        print(data)
