import pandas as pd

# 读取CSV文件
data = pd.read_csv('ground_truth.csv', header=None)

# 将数据保存为XLSX文件
data.to_excel('ground_truth.xlsx', index=False, header=None)