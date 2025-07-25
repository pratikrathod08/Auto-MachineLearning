# Auto-MachineLearning 


## Test version
View at:
https://test.pypi.org/project/Auto-MachineLearning/0.0.6/


## How to use 


### Install test package

```bash
pip install --extra-index-url https://test.pypi.org/simple/ Auto-MachineLearning
```

### Read Data

```bash
from auto_machinelearning.ml.preprocessor.datareader import ReadData

reader = ReadData()
df = reader.read_csv("data.csv")
```

### Describe Data

```bash
from auto_machinelearning.ml.preprocessor.datadescriber import DataDescribe

desc = DataDescribe(df_csv, target_column="price")
desc.summarize()  # Print summary
summary_data = desc.get_summary_dict()
```

### Preprocess Data
```bash

from auto_machinelearning.ml.preprocessor.preprocess import AutoPreprocess

# Create the processor instance
processor = AutoPreprocess(
    path="data.csv",                                        # Path to the file
    file_type="csv",                                        # "csv" or "excel"
    store_path="output/processed.csv",                      # Path to save the result
    target_column="price",                                  # Target column name
    preprocessor_store_path="output/preprocessor.pkl",      # path to store preprocessor 
    encoding="utf-8",                                       # Optional parameters passed to read_csv
    delimiter=","                                           # Optional (if CSV uses different delimiter)
)

# Run the full processing pipeline
processed_df= processor.process()

# You can now work with processed_df
print(processed_df.head())
```

### Use stored preprocessor

```bash 
import pandas as pd 
import joblib 

preprocessor = joblib.load("output/preprocessor.pkl")
csv_path = "data/Bengaluru_House_Data.csv"
df = pd.read_csv(csv_path)
preprocessed_df = preprocessor.transform(df)
preprocessed_df = pd.DataFrame(preprocessed_df)

## Note : add headers and output column in processed df as it is returning processed raw data. 
```

### Train Model

```bash 
from auto_machinelearning.ml.regression import AutoRegression

regression = AutoRegression(path="processed.csv", filetype="csv", target_column="price")
result = regression.train_model()
print(result)

```

### Models available 

- Linear Regression
- Ridge Regression 
- Decision Tree
- Random Forest
- Gradiant Boosting

### result(parameters get in result)

- best_model_name
- model
- model_params
- r2_score
- rmse
- mse

### how to store and use trained model

```bash

import joblib 
import numpy as np
import pandas as pd 
from sklearn.metrics import r2_score

## Store trained model to local system
model = result.get("model")
joblib.dump(model, "model.pkl")

## Load model
model = joblib.load("model.pkl")

## Import data and predict
df = pd.read_csv("processed.csv")
X = df.drop(columns=['target_column'])
y = df["target_column"]

y_pred = model.predict(X)

## Evaluate model 
print("r2_score : ", r2_score(y_pred, y))
print("mse : ", mean_squared_error(y_pred, y))
print("rmse : ", np.sqrt(mean_squared_error(y_pred, y)))

```