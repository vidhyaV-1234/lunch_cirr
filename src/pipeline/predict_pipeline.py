import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=r'artifacts\model.pkl'
            preprocessor_path=r'artifacts\proprocessor.pkl'
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



import pandas as pd

class CustomData:
    def __init__(self, 
                 ID: int,
                 N_Days: int,
                 Drug: int,
                 Age: int,
                 Sex: int,
                 Ascites: int,
                 Hepatomegaly: int,
                 Spiders: int,
                 Edema: int,
                 Bilirubin: float,
                 Cholesterol: float,
                 Albumin: float,
                 Copper: float,
                 Alk_Phos: float,
                 SGOT: float,
                 Tryglicerides: float,
                 Platelets: float,
                 Prothrombin: float,
                 Stage: float):
        
        self.ID = ID
        self.N_Days = N_Days
        self.Drug = Drug
        self.Age = Age
        self.Sex = Sex
        self.Ascites = Ascites
        self.Hepatomegaly = Hepatomegaly
        self.Spiders = Spiders
        self.Edema = Edema
        self.Bilirubin = Bilirubin
        self.Cholesterol = Cholesterol
        self.Albumin = Albumin
        self.Copper = Copper
        self.Alk_Phos = Alk_Phos
        self.SGOT = SGOT
        self.Tryglicerides = Tryglicerides
        self.Platelets = Platelets
        self.Prothrombin = Prothrombin
        self.Stage = Stage
    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "ID": [self.ID],
                "N_Days": [self.N_Days],
                "Drug": [self.Drug],
                "Age": [self.Age],
                "Sex": [self.Sex],
                "Ascites": [self.Ascites],
                "Hepatomegaly": [self.Hepatomegaly],
                "Spiders": [self.Spiders],
                "Edema": [self.Edema],
                "Bilirubin": [self.Bilirubin],
                "Cholesterol": [self.Cholesterol],
                "Albumin": [self.Albumin],
                "Copper": [self.Copper],
                "Alk_Phos": [self.Alk_Phos],
                "SGOT": [self.SGOT],
                "Tryglicerides": [self.Tryglicerides],
                "Platelets": [self.Platelets],
                "Prothrombin": [self.Prothrombin],
                "Stage": [self.Stage],
            }

    

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)