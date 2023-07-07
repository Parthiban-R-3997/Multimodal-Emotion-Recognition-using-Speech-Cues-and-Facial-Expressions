import streamlit as st
from src.pipeline.predict_pipeline import Prediction_NLP, Prediction_CV
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import cv2

prediction_nlp = Prediction_NLP()
prediction_cv = Prediction_CV()


RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


class MyVideoTransformer(VideoTransformerBase):

    def transform(self, frame):
        # Perform your image processing here
        img = frame.to_ndarray(format="bgr24")
        processed_frame = prediction_cv.image_prediction(img)
        return processed_frame



def run_streamlit_app():
    # Set page config to a clean and minimal style
   # Set page config to a clean and minimal style
    st.set_page_config(
        layout="centered",
        page_title="Speech and Facial Emotion Detection",
        page_icon="😃"
        )

    # Set title and description with altered text size and color
    st.markdown("<span style='font-size:40px; color:blue;'>Speech Text and Facial Emotion Detection</span>",unsafe_allow_html=True)
    st.markdown("<span style='font-size:16px; color:purple;'>This web application uses speech text and facial expression to predict emotions.</span>", unsafe_allow_html=True)
    
    # Create a column layout
    col1, col2 = st.columns(2)

    # Create a microphone button in the left column for audio processing
    with col1:
        if st.button("Start Audio Recording"):
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
            #if st.button("Start Camera"):
            # Start the video stream capture
            webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, 
                            rtc_configuration=RTC_CONFIGURATION,
                              video_processor_factory=MyVideoTransformer)

            
if __name__ == "__main__":
    run_streamlit_app()
