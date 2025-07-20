import os, sys 
import pandas as pd 
import numpy as np 

from auto_machinelearning.ml.regression.entity.artifacts_entity import DataIngestionArtifacts
from auto_machinelearning.ml.regression.entity.artifacts_entity import DataTransformationArtifacts

class DataTransformation: 
    def __init__(self, data_ingestion_artifacts: DataIngestionArtifacts, target_column: str): 
        self.train_df = data_ingestion_artifacts.train_df.copy()
        self.test_df = data_ingestion_artifacts.test_df.copy()
        self.target_column = target_column

    def run_data_transformation(self) -> DataTransformationArtifacts:
        # Separate X and y
        X_train = self.train_df.drop(columns=[self.target_column])
        y_train = self.train_df[[self.target_column]]
        X_test = self.test_df.drop(columns=[self.target_column])
        y_test = self.test_df[[self.target_column]]

        # # Apply preprocessing on features
        # preprocessor = AutoPreprocessor(X_train, target_column=None)  # no target needed for X
        # X_train_processed = preprocessor.run_preprocessing()
        # X_test_processed = preprocessor.scale_dataframe(preprocessor.convert_cat_to_num(X_test))

        # Return all in artifacts
        return DataTransformationArtifacts(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )