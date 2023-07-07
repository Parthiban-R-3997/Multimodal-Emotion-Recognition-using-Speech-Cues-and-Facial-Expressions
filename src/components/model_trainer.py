import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation_nlp, DataTransformation_cv
from src.utils import model_training_nlp, model_training_cv, save_model
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Conv2D, MaxPooling2D, Flatten
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

@dataclass
class ModelTrainerConfig_nlp:
    trained_model_file_path_nlp = os.path.join("artifacts", "Bi-LSTM.h5")


class ModelTrainer_nlp:
    def __init__(self, X_train_final, y_train_final, X_valid_final, y_valid_final, X_test_final, y_test_final,
                 train_max_sequence_length_nlp, train_voc_size_nlp):
        self.model_trainer_config_nlp = ModelTrainerConfig_nlp()
        self.X_train_final = X_train_final
        self.y_train_final = y_train_final
        self.X_valid_final = X_valid_final
        self.y_valid_final = y_valid_final
        self.X_test_final = X_test_final
        self.y_test_final = y_test_final
        self.voc_size = train_voc_size_nlp
        self.max_sequence_length = train_max_sequence_length_nlp

    def initiate_model_trainer_nlp(self):
        logging.info('Model Trainer initialization started')
        try:
            embedding_vector_features = 300  # Feature representation
            model_nlp = Sequential()
            model_nlp.add(Embedding(self.voc_size, embedding_vector_features, input_length=self.max_sequence_length))
            model_nlp.add(Bidirectional(LSTM(512, dropout=0.3, return_sequences=True)))
            model_nlp.add(Bidirectional(LSTM(256, dropout=0.3, return_sequences=True)))
            model_nlp.add(Bidirectional(LSTM(128, dropout=0.3)))
            model_nlp.add(Dense(6, activation='sigmoid'))

            optimizer = optimizers.Adam(lr=0.005)
            model_nlp.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            model_nlp.summary()
            logging.info("Model initialized successfully")

            # Define the callback
            callback = EarlyStopping(monitor="val_loss", patience=4)

            # Call the model_training_nlp function from utils.py
            history_nlp = model_training_nlp(X_train=self.X_train_final, y_train=self.y_train_final,
                                             validation_data=(self.X_valid_final, self.y_valid_final),
                                             epochs=1, batch_size=128, callback=callback, model=model_nlp)

            save_model(
                file_path=self.model_trainer_config_nlp.trained_model_file_path_nlp,
                obj=model_nlp)

            return history_nlp

        except Exception as e:
            logging.error(f"Error occurred during model training: {e}")
            raise CustomException(e, sys)


@dataclass
class ModelTrainerConfig_cv:
    trained_model_file_path_cv = os.path.join("artifacts", "CNN_final.h5")


class ModelTrainer_cv:
    def __init__(self, train_generator_cv, test_generator_cv, input_shape):
        self.model_trainer_config_cv = ModelTrainerConfig_cv()
        self.train_generator_cv = train_generator_cv
        self.test_generator_cv = test_generator_cv
        self.input_shape = input_shape

    def initiate_model_trainer_cv(self):
        logging.info('Model Trainer initialization started')
        try:
            model_cv = Sequential()
            # Convolutional layers
            model_cv.add(Conv2D(256, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
            model_cv.add(MaxPooling2D(pool_size=(2, 2)))
            model_cv.add(Dropout(0.3))

            model_cv.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
            model_cv.add(MaxPooling2D(pool_size=(2, 2)))
            model_cv.add(Dropout(0.3))

            model_cv.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
            model_cv.add(MaxPooling2D(pool_size=(2, 2)))
            model_cv.add(Dropout(0.3))

            model_cv.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
            model_cv.add(MaxPooling2D(pool_size=(2, 2)))
            model_cv.add(Dropout(0.3))

            model_cv.add(Flatten())
            # Fully connected layers
            model_cv.add(Dense(512, activation='relu'))
            model_cv.add(Dropout(0.3))
            model_cv.add(Dense(512, activation='relu'))
            model_cv.add(Dropout(0.3))
            model_cv.add(Dense(256, activation='relu'))
            model_cv.add(Dropout(0.3))

            # Output layer
            model_cv.add(Dense(7, activation='softmax'))

            optimizer = optimizers.Adam(lr=0.0005)
            model_cv.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            print(model_cv.summary())
            logging.info("Model initialized successfully")

            callback =EarlyStopping(monitor='val_acc', patience=10)

            # Call the model_training_cv function from utils.py
            history_cv = model_training_cv(train_generator_cv=self.train_generator_cv,
                                           steps_per_epoch=self.train_generator_cv.samples // self.train_generator_cv.batch_size,
                                           epochs=1,
                                           validation_data=self.test_generator_cv,
                                           validation_steps=self.test_generator_cv.samples // self.test_generator_cv.batch_size,
                                           callback=callback,
                                           model=model_cv)

            save_model(
                file_path=self.model_trainer_config_cv.trained_model_file_path_cv,
                obj=model_cv)

            return history_cv

        except Exception as e:
            logging.error(f"Error occurred during model training: {e}")
            raise CustomException(e, sys)
