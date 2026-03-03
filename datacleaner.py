import pandas as pd
import numpy as np
import json
import os

class SmartDataCleaner:
    def __init__(self, file_path, config_path):
        self.file_path = file_path
        self.config = self.load_config(config_path)
        self.df = pd.read_csv(file_path)
        self.log = []
    
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def standardize_columns(self):
        self.df.columns = (
            self.df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )
        self.log.append("Column names standardized.")
    
    def handle_missing_values(self):
        threshold = self.config["missing_threshold"]
        initial_rows = len(self.df)
        
        for col in self.df.columns:
            missing_ratio = self.df[col].isnull().mean()
            
            if missing_ratio > threshold:
                self.df.drop(columns=[col], inplace=True)
                self.log.append(f"Dropped column '{col}' due to high missing ratio.")
            else:
                if self.df[col].dtype == "object":
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                    self.log.append(f"Filled missing categorical '{col}' with mode.")
                else:
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                    self.log.append(f"Filled missing numeric '{col}' with median.")
        
        self.log.append(f"Rows before cleaning: {initial_rows}")
        self.log.append(f"Rows after missing handling: {len(self.df)}")
    
    def fix_data_types(self):
        for col in self.df.columns:
            if "date" in col:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
                self.log.append(f"Converted '{col}' to datetime.")
    
    def remove_duplicates(self):
        if self.config["drop_duplicates"]:
            before = len(self.df)
            self.df.drop_duplicates(inplace=True)
            after = len(self.df)
            self.log.append(f"Removed {before - after} duplicate rows.")
    
    def remove_outliers_iqr(self):
        numeric_cols = self.df.select_dtypes(include=np.number).columns
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            before = len(self.df)
            self.df = self.df[
                (self.df[col] >= Q1 - 1.5 * IQR) &
                (self.df[col] <= Q3 + 1.5 * IQR)
            ]
            after = len(self.df)
            
            if before != after:
                self.log.append(f"Removed {before - after} outliers from '{col}'.")
    
    def save_output(self):
        # Create output directory if it doesn't exist
        os.makedirs("output", exist_ok=True)
        
        self.df.to_csv("output/cleaned_data.csv", index=False)
        
        with open("output/cleaning_report.txt", "w") as f:
            for entry in self.log:
                f.write(entry + "\n")
    
    def run_pipeline(self):
        self.standardize_columns()
        self.handle_missing_values()
        self.fix_data_types()
        self.remove_duplicates()
        
        if self.config["outlier_method"] == "iqr":
            self.remove_outliers_iqr()
        
        self.save_output()
        print("Cleaning Completed Successfully!")