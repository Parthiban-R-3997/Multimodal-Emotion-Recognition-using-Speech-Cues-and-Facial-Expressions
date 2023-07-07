import sys
import pandas as pd
import speech_recognition as sr
from src.exception import CustomException
from src.logger import logging
from src.utils import load_model_nlp, load_model_cv
from src.components.data_transformation import  DataTransformation_test_data_nlp
import cv2
import mediapipe as mp
import numpy as np



class Prediction_NLP:
       
   def audio_recording_prediction(self):
        try:
          # Set the initial flag value
          record_flag = False

          # Set up speech recognition
          recognizer = sr.Recognizer()

          
          while True:
                with sr.Microphone() as source:
                    print("Listening...")
                    audio = recognizer.listen(source)
                
                try:
                    # Perform speech recognition on the audio
                    result = recognizer.recognize_google(audio)
                    print("Recognized speech:", result)
                    class_names = {
                                0: 'sadness',
                                1: 'joy',
                                2: 'love',
                                3: 'anger',
                                4: 'fear',
                                5: 'surprise'
                            }
                    model_path= 'artifacts\Bi-LSTM.h5'
                    nlp_model_load = load_model_nlp(model_path)
                    data_transformer = DataTransformation_test_data_nlp() 
                    preprocessed_texts = data_transformer.returning_preprocessed_text(result) 
                    prediction = nlp_model_load.predict(preprocessed_texts)
                    predicted_class_index = prediction.argmax(axis=1)[0]
                    predicted_class_name = class_names[predicted_class_index]
                    predicted_probabilities = prediction[0]
                    #print(predicted_class_name, predicted_probabilities, class_names, result)
                    return predicted_class_name, predicted_probabilities,class_names,result 



                except sr.UnknownValueError:
                    print("No speech detected.")

        except KeyboardInterrupt:
            print("Listening interrupted.")  
             
       

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
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)

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
                                (255, 0, 0), 2, cv2.LINE_AA)

            return frame

        except Exception as e:
            logging.error('Error occurred while trying to capture the image')
            raise CustomException(e, sys)

        
