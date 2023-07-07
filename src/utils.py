import os
import sys

import numpy as np 
import pandas as pd

from keras.callbacks import EarlyStopping
from keras.models import load_model


from src.exception import CustomException
from src.logger import logging

def model_training_nlp(X_train_final, y_train_final, validation_data,
                                epochs, batch_size, callback, model):
    try:
        logging.info("Model training has started...")

        history_nlp = model.fit(X_train_final, y_train_final, validation_data=validation_data, epochs=epochs,
                    batch_size=batch_size,callbacks=[callback])
        
        logging.info("Model training has completed...")

        return history_nlp
    


    except Exception as e:
        logging.error(f"Error occurred during model training: {e}")
        raise CustomException(e, sys)
    
def model_training_cv(train_generator_cv, steps_per_epoch, epochs,
                                validation_data, validation_steps, callback, model):
    try:
        logging.info("Model training has started...")

        history_cv = model.fit_generator(train_generator_cv, 
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs, 
                            validation_data=validation_data,
                            validation_steps=validation_steps,
                            callbacks=[callback])
        
        logging.info("Model training has completed...")

        return history_cv
    
    except Exception as e:
        logging.error(f"Error occurred during model training: {e}")
        raise CustomException(e, sys)
    


def save_model(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        obj.save(file_path)
        logging.info("Model has saved")

    except Exception as e:
        logging.error(f"Error occurred while saving the model: {e}")
        raise CustomException(e, sys)    
    
def load_model_nlp(file_path):

    try:
        model_nlp = load_model(file_path)
        logging.info("NLP Model loaded successfully")
        return model_nlp
    
    except Exception as e:
        logging.error(f"Error occurred while loading the NLP model: {e}")
        raise CustomException(e, sys)


def load_model_cv(file_path):

    try:
        model_cv = load_model(file_path)
        logging.info("CV Model loaded successfully")
        return model_cv
    
    except Exception as e:
        logging.error(f"Error occurred while loading the CV model: {e}")
        raise CustomException(e, sys)
        