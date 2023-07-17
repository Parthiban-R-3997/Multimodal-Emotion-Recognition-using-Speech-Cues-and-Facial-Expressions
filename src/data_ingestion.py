import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from src.components.data_transformation import  DataTransformation_nlp,DataTransformation_cv
from src.components.model_trainer import ModelTrainer_nlp, ModelTrainer_cv
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train_nlp.csv")
    test_data_path: str = os.path.join('artifacts', "test_nlp.csv")
    validation_data_path: str = os.path.join('artifacts', "validation_nlp.csv")
    train_cv_folder_path: str = os.path.join('artifacts', 'train_cv')
    test_cv_folder_path: str = os.path.join('artifacts', 'test_cv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Create artifacts folder if it doesn't exist
            os.makedirs('artifacts', exist_ok=True)

            # Read the train_nlp.csv file and save it to artifacts folder
            df = pd.read_csv('notebook/data/training_nlp.csv')
            df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info("Saved train_nlp.csv to artifacts folder")

            # Read the test_nlp.csv file and save it to artifacts folder
            df = pd.read_csv('notebook/data/test_nlp.csv')
            df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Saved test_nlp.csv to artifacts folder")

            # Read the validation_nlp.csv file and save it to artifacts folder
            df = pd.read_csv('notebook/data/validation_nlp.csv')
            df.to_csv(self.ingestion_config.validation_data_path, index=False, header=True)
            logging.info("Saved validation_nlp.csv to artifacts folder")

            # Create train_cv and test_cv folders under artifacts folder
            os.makedirs(self.ingestion_config.train_cv_folder_path, exist_ok=True)
            os.makedirs(self.ingestion_config.test_cv_folder_path, exist_ok=True)
            logging.info("Created train_cv and test_cv folders under artifacts folder")

            # Move the subfolders and files from notebook/data/train_cv to artifacts/train_cv
            train_cv_path = os.path.join('notebook/data', 'train_cv')
            for folder_name in os.listdir(train_cv_path):
                folder_path = os.path.join(train_cv_path, folder_name)
                if os.path.isdir(folder_path):
                    target_folder_path = os.path.join(self.ingestion_config.train_cv_folder_path, folder_name)
                    os.makedirs(target_folder_path, exist_ok=True)
                    for file_name in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, file_name)
                        target_file_path = os.path.join(target_folder_path, file_name)
                        os.rename(file_path, target_file_path)
                    logging.info(f"Moved subfolder '{folder_name}' from notebook/data/train_cv to artifacts/train_cv")

            # Move the subfolders and files from notebook/data/test_cv to artifacts/test_cv
            test_cv_path = os.path.join('notebook/data', 'test_cv')
            for folder_name in os.listdir(test_cv_path):
                folder_path = os.path.join(test_cv_path, folder_name)
                if os.path.isdir(folder_path):
                    target_folder_path = os.path.join(self.ingestion_config.test_cv_folder_path, folder_name)
                    os.makedirs(target_folder_path, exist_ok=True)
                    for file_name in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, file_name)
                        target_file_path = os.path.join(target_folder_path, file_name)
                        os.rename(file_path, target_file_path)
                    logging.info(f"Moved subfolder '{folder_name}' from notebook/data/test_cv to artifacts/test_cv")

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.validation_data_path,
                self.ingestion_config.train_cv_folder_path,
                self.ingestion_config.test_cv_folder_path
            )
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data_path, test_data_path, validation_data_path, train_cv_folder_path, test_cv_folder_path = obj.initiate_data_ingestion()
    
    # Perform NLP data preprocessing
    #data_transformation_nlp = DataTransformation_nlp(train_data_path, test_data_path, validation_data_path)
    #train_data_nlp, train_labels_nlp, train_max_sequence_length_nlp, train_voc_size_nlp = data_transformation_nlp.preprocess_dataset(train_data_path)
    #test_data_nlp, test_labels_nlp, test_max_sequence_length_nlp, test_voc_size_nlp = data_transformation_nlp.preprocess_dataset(test_data_path)
    #validation_data_nlp, validation_labels_nlp, validation_max_sequence_length_nlp, validation_voc_size_nlp = data_transformation_nlp.preprocess_dataset(validation_data_path)
    #X_train_final,y_train_final= data_transformation_nlp.convert_to_arrays(train_data_nlp, train_labels_nlp)
    #X_valid_final,y_valid_final= data_transformation_nlp.convert_to_arrays(validation_data_nlp, validation_labels_nlp)
    #X_test_final,y_test_final= data_transformation_nlp.convert_to_arrays(test_data_nlp, test_labels_nlp)
    

    ## Perform Model Initialization on NLP
    #model_trainer_nlp = ModelTrainer_nlp(X_train_final,y_train_final,X_valid_final,y_valid_final,X_test_final,y_test_final,train_max_sequence_length_nlp, train_voc_size_nlp)
    #model_trainer_nlp.initiate_model_trainer_nlp()
    

    # Perform CV data preprocessing
    #data_transformation_cv = DataTransformation_cv(train_cv_folder_path, test_cv_folder_path)
    #train_generator_cv, test_generator_cv,input_shape = data_transformation_cv.preprocess_dataset()


    ## Perform Model Initialization on CV
    #model_trainer_cv = ModelTrainer_cv(train_generator_cv, test_generator_cv,input_shape)
    #model_trainer_cv.initiate_model_trainer_cv()


    

