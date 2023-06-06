import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
import numpy as np


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 LIMIT_BAL:float,
                 SEX:int,
                 EDUCATION:int,
                 MARRIAGE:int,
                 AGE:float,
                 PAY_1:int,
                 BILL_AMT1:float,
                 PAY_AMT1:float,
                 PAY_AMT2:float,
                 PAY_AMT3:float,
                 PAY_AMT4:float,
                 PAY_AMT5:float,
                 PAY_AMT6:float):
        
        self.LIMIT_BAL=LIMIT_BAL
        self.SEX=SEX
        self.EDUCATION=EDUCATION
        self.MARRIAGE=MARRIAGE
        self.AGE=AGE
        self.PAY_1=PAY_1
        self.BILL_AMT1 = BILL_AMT1
        self.PAY_AMT1 = PAY_AMT1
        self.PAY_AMT2 = PAY_AMT2
        self.PAY_AMT3 = PAY_AMT3
        self.PAY_AMT4 = PAY_AMT4
        self.PAY_AMT5 = PAY_AMT5
        self.PAY_AMT6 = PAY_AMT6

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'LIMIT_BAL':[self.LIMIT_BAL],
                'SEX':[self.SEX],
                'EDUCATION':[self.EDUCATION],
                'MARRIAGE':[self.MARRIAGE],
                'AGE':[self.AGE],
                'PAY_1':[self.PAY_1],
                'BILL_AMT1':[self.BILL_AMT1],
                'PAY_AMT1':[self.PAY_AMT1],
                'PAY_AMT2':[self.PAY_AMT2],
                'PAY_AMT3':[self.PAY_AMT3],
                'PAY_AMT4':[self.PAY_AMT4],
                'PAY_AMT5':[self.PAY_AMT5],
                'PAY_AMT6':[self.PAY_AMT6]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)

            