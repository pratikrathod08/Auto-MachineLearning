import os, sys 

import numpy as np 
import pandas as pd 

from auto_machinelearning.ml.preprocessor.datareader import ReadData
from auto_machinelearning.ml.preprocessor.datadescriber import DataDescribe
from auto_machinelearning.ml.preprocessor.auto_preprocess import AutoPreprocessor

class AutoPreprocess: 
    def __init__(self, path: str, file_type: str, store_path: str, target_column: str, **params): 
        self.path = path
        self.file_type = file_type
        self.store_path = store_path
        self.target_column = target_column
        self.params = params
        self.df = None
        self.description = None
        self.preprocessed_df = None

    def read_file(self):
        read_data = ReadData()
        if self.file_type == "csv": 
            self.df = read_data.read_csv(self.path, **self.params)
        elif self.file_type == "excel": 
            self.df = read_data.read_excel(self.path, **self.params)
        return self.df
    
    def describe_data(self, df: pd.DataFrame, target_column: str):
        data_describer = DataDescribe(df, target_column)
        data_describer.summarize()
        description = data_describer.get_summary_dict()
        return description
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str): 
        data_processor = AutoPreprocessor(df, target_column)
        processed_df = data_processor.run_preprocessing()
        return processed_df
    
    def store_df(self): 
        if self.preprocessed_df is not None:
            dir_path = os.path.dirname(self.store_path)
            os.makedirs(dir_path, exist_ok=True)
            self.preprocessed_df.to_csv(self.store_path, index=False)
        else:
            raise ValueError("Preprocessed dataframe is None.")

    def process(self): 
        self.df = self.read_file()
        self.description = self.describe_data(self.df, self.target_column)
        self.preprocessed_df = self.preprocess_data(self.df, self.target_column)
        self.store_df()
        return self.preprocessed_df
