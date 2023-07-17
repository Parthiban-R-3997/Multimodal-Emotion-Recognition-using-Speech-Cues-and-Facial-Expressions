import os
import sys
import pandas as pd
import numpy as np
import nltk
import speech_recognition as sr
import pyaudio
import cv2
import mediapipe as mp
import streamlit as st
import assemblyai as aai
from src.exception import CustomException
from src.logger import logging
from src.utils import load_model_nlp, load_model_cv
from src.data_transformation import  DataTransformation_test_data_nlp
from audio_recorder_streamlit import audio_recorder



class Prediction_NLP:
    def __init__(self):
        # Your API token is already set here
        aai.settings.api_key = "e3f336e6533b41588c90f4dbf56c317c"

        # Create a transcriber object.
        self.transcriber = aai.Transcriber()

          # Initialize the data transformer
        self.data_transformer = DataTransformation_test_data_nlp()

    def AudioRecorderApp(self):
        audio_bytes = st.markdown("<h5 style='color: red;'>Start Audio Recording</h5>", unsafe_allow_html=True)
        audio_bytes = audio_recorder()

        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")

            # Save audio to a temporary file
            audio_file = "audio.wav"
            with open(audio_file, "wb") as f:
                f.write(audio_bytes)

            # Transcribe the audio file
            transcript = self.transcriber.transcribe(audio_file)
            transcribed_text = transcript.text
            st.write(transcribed_text)

            # Perform emotion prediction only if recognized_text is not empty
            if transcribed_text:
                try:
                    class_names = {
                        0: 'sadness 🥺',
                        1: 'joy 😃',
                        2: 'love 🥰',
                        3: 'anger 😠',
                        4: 'fear 😨',
                        5: 'surprise 😵'
                    }
                    model_path = 'artifacts/Bi-LSTM.h5'
                    nlp_model_load = load_model_nlp(model_path)
                    
                    preprocessed_texts = self.data_transformer.returning_preprocessed_text(transcribed_text)
                    prediction = nlp_model_load.predict(preprocessed_texts)
                    predicted_class_index = prediction.argmax(axis=1)[0]
                    predicted_class_name = class_names[predicted_class_index]
                    st.write(f"Predicted sentiment: {predicted_class_name}")
                    predicted_probabilities = prediction[0]
                    
                    
                    # Create a DataFrame with predicted sentiment and probabilities
                    result_df = pd.DataFrame({
                        'Sentiment': [class_names[class_index] for class_index in class_names],
                        'Probability': ['%.6f' % prob for prob in predicted_probabilities]
                    })

                    # Sort the DataFrame by probabilities in ascending order
                    result_df = result_df.sort_values(by='Probability', ascending=False)

                   
                    # Display the predicted sentiment and probabilities in a table
                    st.markdown("<h4 style='font-size: 16px; color: purple;'>Prediction Results:</h4>", unsafe_allow_html=True)
                    st.table(result_df)

                except Exception as e:
                    st.error("Error occurred during emotion prediction:")
                    st.error(e)
          
class Prediction_CV:
   
    def image_prediction(self, frame):
        try:
            emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}
            
            mp_face_mesh = mp.solutions.face_mesh
            # Create an instance of FaceMesh
            face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=10, min_detection_confidence=0.6)

            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run face mesh on the frame using MediaPipe
            results = face_mesh.process(frame_rgb)
            
            # Check if any faces are detected
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Convert normalized landmarks to pixel coordinates
                    h, w, _ = frame.shape
                    landmarks = []
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        landmarks.append((x, y))

                    # Approximate the bounding box of the face using landmarks
                    x, y, w, h = cv2.boundingRect(np.array(landmarks))

                    # Draw bounding box on the frame (in BGR format)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (11, 221, 221), 2)

                    # Extract the face region from the frame
                    face_img = frame[y:y+h, x:x+w]

                    # Check if the face region is empty or has a size of zero
                    if face_img.size == 0:
                        continue

                    # Convert the face image to grayscale
                    gray_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

                    # Resize the grayscale image to match the expected input shape of the model
                    resized_img = cv2.resize(gray_img, (48, 48))

                    # Expand dimensions to create a batch dimension
                    input_img = np.expand_dims(resized_img, axis=-1)

                    # Normalize the input image
                    input_img = input_img / 255.0

                    # Load the saved model
                    model_path = 'artifacts/CNN_final.h5'
                    cv_model_load = load_model_cv(model_path)

                    # Predict the emotion
                    emotion_prediction = cv_model_load.predict(np.expand_dims(input_img, axis=0))
                    maxindex = int(np.argmax(emotion_prediction))
                    cv2.putText(frame, emotion_dict[maxindex], (x + 5, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (58, 221, 55), 2, cv2.LINE_AA)

            return frame

        except Exception as e:
            logging.error('Error occurred while trying to capture the image')
            raise CustomException(e, sys)

        
