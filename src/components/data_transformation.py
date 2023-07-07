import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
import nltk
import tensorflow.compat.v1 as tf
from tensorflow.keras.preprocessing.text import one_hot
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

class DataTransformation_nlp:
   def __init__(self, train_data_path, test_data_path, validation_data_path):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.validation_data_path = validation_data_path

        # Initialize the stop words
        self.stop_words = set(stopwords.words('english'))   

        # Initialize the PorterStemmer
        self.ps = PorterStemmer()

   def preprocess_text(self, texts):
        logging.info("Performing text preprocessing...")
        corpus = []
        for text in texts:
            review = re.sub('[^a-zA-Z]', ' ', text.lower())
            review = review.split()
            review = [self.ps.stem(word) for word in review if word not in self.stop_words]
            review = ' '.join(review)
            corpus.append(review)

        return corpus

   def preprocess_dataset(self, filename):
        try:
            logging.info("Loading dataset...")
            data = pd.read_csv(filename)
            texts = data['text'].values
            labels = data['label'].values

            # Preprocess the text data
            preprocessed_texts = self.preprocess_text(texts)

            # Convert text to sequences
            voc_size = 5000
            onehot_repr = [one_hot(words, voc_size) for words in preprocessed_texts]

            # Pad sequences for uniform length
            max_sequence_length = 200
            padded_sequences = pad_sequences(onehot_repr, padding='pre', maxlen=max_sequence_length)

            # Convert labels to one-hot encoded vectors
            num_classes = len(set(labels))
            labels = tf.keras.utils.to_categorical(labels, num_classes)

            logging.info("Data preprocessing completed.")

            return padded_sequences, labels, max_sequence_length, voc_size

        except Exception as e:
            logging.error(f"Error occurred during data preprocessing: {e}")
            raise CustomException(e, sys)

   def convert_to_arrays(self, padded_sequences, labels):
        try:
            logging.info("Converting texts and labels to arrays")
            padded_sequences_array = np.array(padded_sequences)
            labels_array = np.array(labels)
            logging.info("Conversion completed successfully")
            return padded_sequences_array, labels_array
        except Exception as e:
            logging.error(f"Error occurred during conversion to array format: {e}")
            raise CustomException(e, sys)
   

class DataTransformation_test_data_nlp:   
   def preprocess_text_for_prediction(self,text):
        try:
            # Preprocess the text
            processed_text = text.lower()
            processed_text = re.sub('[^a-zA-Z]', ' ', processed_text)
            processed_text = processed_text.split()
            ps = PorterStemmer()
            processed_text = [ps.stem(word) for word in processed_text if word not in stopwords.words('english')]
            processed_text = ' '.join(processed_text)

            return processed_text
        
        except Exception as e:
            raise CustomException(e, sys)
        

   def returning_preprocessed_text(self,result):
        try:
            preprocessed_text = self.preprocess_text_for_prediction(result)

            # Convert text to one-hot encoding
            voc_size = 5000
            onehot_repr = [one_hot(preprocessed_text, voc_size)]

            # Pad sequence for uniform length
            max_sequence_length = 200
            padded_sequence = pad_sequences(onehot_repr, padding='pre', maxlen=max_sequence_length)

            return padded_sequence
        
        except Exception as e:
            raise CustomException(e, sys)
       
       
class DataTransformation_cv:
    def __init__(self, train_dir, test_dir):
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.img_width, self.img_height = 48, 48
        self.input_shape = (48, 48, 1)

    def preprocess_dataset(self):
        logging.info("Entered the CV data preprocessing method")
        try:
            # Set up the data generators
            train_datagen = ImageDataGenerator(rescale=1.0/255,
                                               shear_range=0.2,
                                               zoom_range=0.2,
                                               horizontal_flip=True)
            test_datagen = ImageDataGenerator(rescale=1.0/255)

            train_generator = train_datagen.flow_from_directory(self.train_dir,
                                                                target_size=(self.img_width, self.img_height),
                                                                batch_size=128,
                                                                color_mode="grayscale",
                                                                class_mode='categorical')

            test_generator = test_datagen.flow_from_directory(self.test_dir,
                                                              target_size=(self.img_width, self.img_height),
                                                              batch_size=128,
                                                              color_mode="grayscale",
                                                              class_mode='categorical')

            logging.info("CV data preprocessing completed")

            return (train_generator, test_generator, self.input_shape)
        except Exception as e:
            raise CustomException(e, sys)
