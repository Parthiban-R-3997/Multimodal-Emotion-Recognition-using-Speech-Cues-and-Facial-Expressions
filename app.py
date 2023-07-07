import streamlit as st
import cv2
from src.pipeline.predict_pipeline import Prediction_NLP, Prediction_CV

prediction_nlp = Prediction_NLP()
prediction_cv = Prediction_CV()

def main():
    # Set page config to a clean and minimal style
    st.set_page_config(layout="wide", page_title="Speech and Facial Emotion Detection", page_icon="🎙️")

    # Set background color
    page_bg_color = "#93c47d"
    st.markdown(f"""
        <style>
        body {{
            background-color: {page_bg_color};
        }}
        .reportview-container .main .block-container {{
            max-width: 1200px;
            padding: 2rem;
        }}
        </style>
        """, unsafe_allow_html=True)

    # Set title and description
    st.title("Speech and Facial Emotion Detection")
    st.markdown("This app uses speech recognition and facial expression recognition to predict emotions.")

    # Create a column layout
    col1, col2 = st.columns(2)

    # Create a microphone button in the left column for audio processing
    with col1:
        if st.button("Start Recording"):
            # Call the audio_recording_prediction function when the button is clicked
            predicted_class, predicted_probabilities, class_names, results = prediction_nlp.audio_recording_prediction()

            # Display the recognized speech text
            st.subheader("Recognized Speech:")
            st.write(results)

            # Display the predicted sentiment and probabilities
            st.subheader("Prediction Results:")
            st.write(f"Predicted Sentiment: {predicted_class}")
            st.write("Prediction Probabilities:")
            for class_index, class_name in class_names.items():
                probability = predicted_probabilities[class_index]
                st.write(f"{class_name}: {probability}")

    # Create a camera button in the right column for video processing
    with col2:
        if st.button("Start Camera"):
            # Start the video stream capture
            cap = cv2.VideoCapture(0)

            # Create an instance of FaceMesh
            face_mesh = prediction_cv.get_face_mesh()

            # Read and display video frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Call the image_prediction function to perform emotion detection on the current frame
                frame_with_emotion = prediction_cv.image_prediction(frame, face_mesh)

                # Display the video stream
                st.subheader("Emotion Detection:")
                st.video(frame_with_emotion, format="BGR")

                # Break the loop if 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Release the video capture and destroy any OpenCV windows
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
