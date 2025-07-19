# Auto-MachineLearning 


## Test version
View at:
https://test.pypi.org/project/Auto-MachineLearning/0.0.1/


## How to use 


`install test package`

```bash
pip install --extra-index-url https://test.pypi.org/simple/ Auto-MachineLearning
```

`Read Data`

```bash
from Auto_MachineLearning.ml.preprocessor.datareader import ReadData

reader = ReadData()
df = reader.read_csv("data.csv")
```

`Describe Data`

```bash
from auto_machinelearning.ml.preprocessor.datadescriber import DataDescribe

desc = DataDescribe(df_csv, target_column="price")
desc.summarize()  # Print summary
summary_data = desc.get_summary_dict()
```

`Preprocess Data`
```bash
from auto_machinelearning.ml.preprocessor.process import AutoPreprocess

#### Create the processor instance
processor = AutoPreprocess(
    path=""                           # Path to the file
    file_type="csv",                  # "csv" or "excel"
    store_path="output/processed.csv",# Where to save the result
    target_column="price",            # Target column name
    encoding="utf-8",                 # Optional parameters passed to read_csv
    delimiter=","                     # Optional (if CSV uses different delimiter)
)

#### Run the full processing pipeline
processed_df = processor.process()

#### You can now work with processed_df
print(processed_df.head())
```