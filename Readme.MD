# Auto-MachineLearning 

## Test version
View at:
https://test.pypi.org/project/Auto-MachineLearning/0.0.1/


## Hou to use 

`Read data`
from Auto_MachineLearning.ml.preprocessor.datareader import ReadData

reader = ReadData()
df = reader.read_csv("data.csv")

`describe data`
from auto_machinelearning.ml.preprocessor.datadescriber import DataDescribe

desc = DataDescribe(df_csv, target_column="price")
desc.summarize()  # Print summary
summary_data = desc.get_summary_dict()