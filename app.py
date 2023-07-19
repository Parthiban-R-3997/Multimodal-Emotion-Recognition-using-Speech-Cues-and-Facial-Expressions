import streamlit as st
from src.predict_pipeline import Prediction_NLP, Prediction_CV
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
from twilio.rest import Client
import os
from dotenv import load_dotenv

# Load environment variables from the "twilio.env" file
load_dotenv(dotenv_path="twilio.env")

prediction_nlp = Prediction_NLP()
prediction_cv = Prediction_CV()

account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
client = Client(account_sid, auth_token)

token = client.tokens.create()


#RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
RTC_CONFIGURATION = RTCConfiguration({"iceServers": token.ice_servers})
 

class MyVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Perform your image processing here
        img = frame.to_ndarray(format="bgr24")
        processed_frame = prediction_cv.image_prediction(img)
        return processed_frame

# Set page config to a clean and minimal style with wide layout
    st.set_page_config(
        layout="wide",
        page_title="Multimodal Speech and Facial Emotion Detection",
        page_icon="😃"
    )


def run_streamlit_app():
    activity = ["Home", "Working Activity"]
    choice = st.sidebar.selectbox("Select Activity", activity)
    st.sidebar.markdown(
    """
    - **Email**: [rparthiban729@gmail.com](mailto:rparthiban729@gmail.com)  
    - **LinkedIn**: [Parthiban Ravichandran](https://www.linkedin.com/in/parthiban-ravichandran/)
    - **GitHub**: [Parthiban-R-3997](https://github.com/Parthiban-R-3997)
    """,
    unsafe_allow_html=True
)


    if choice == "Home":
            
            # Set title and description with altered text size and color
            st.markdown(
                """
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
                    <h1 style="font-size: 30px; color: blue; text-align: center;">Multimodal emotion recognition using Audio Text 🎙️ and Facial image 😃</h1>
                </div>
                """,unsafe_allow_html=True
            )
            
            html_temp_home1 = """<div style="background-color:#e85530;padding:10px">
                                                <h5 style="color:white;text-align:center;">
                                                This web application uses speech text and facial expression to predict emotions.</h5>
                                                </div>
                                                </br>"""
            st.markdown(html_temp_home1, unsafe_allow_html=True)

            

            st.write("""
                    The application has two functionalities.

                    1. Detects text from audio and predicts emotions using Bi-LSTM.

                    2. Detects real time facial emotion recognization using Mediapipe and OpenCV.

                        """)
            


    elif choice == "Working Activity":  

                # Set title and description with altered text size and color
                st.markdown(
                """
                <div style="display: flex; flex-direction: column; align-items: center; justify-content: center;">
                    <h1 style="font-size: 30px; color: blue; text-align: center;">Predicting emotions using NLP 🎙️ and CV techniques 😃</h1>
                </div>
                """,unsafe_allow_html=True
               )  
                        
                # Create a column layout with wider columns
                col1, col2 = st.columns(2)

                # Set the CSS property for the width of the columns
                col1.markdown(
                    f'<style>div.row-widget:nth-child(1) div[role="main"] .element-container {{ width: 600px; }}</style>',
                    unsafe_allow_html=True
                )
                col2.markdown(
                    f'<style>div.row-widget:nth-child(2) div[role="main"] .element-container {{ width: 600px; }}</style>',
                    unsafe_allow_html=True
                )

                # Create a microphone button in the left column for audio processing
                with col1:
                    prediction_nlp.AudioRecorderApp()

                # Create a camera button in the right column for video processing
                with col2:
                    st.markdown("<h5 style='text-align: center; color: green; font-size: 20px;'>Start Video Capture 📸</h5>", unsafe_allow_html=True)
                    # Start the video stream capture
                    webrtc_streamer(
                        key="example",
                        mode=WebRtcMode.SENDRECV,
                        rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=MyVideoTransformer
                    )

    else:
            pass

if __name__ == "__main__":
    run_streamlit_app()
