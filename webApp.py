import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
# import tempfile
import time
from PIL import Image
import os
from PIL import Image
import time
import pickle

# Changing the background to light theme

page_styles = """
    <style>
        [data-testid="stAppViewContainer"]{
            background-color: #e5e5f7;
        }
        [data-testid="stSidebar"]{
            background-color: #e5e5f7;
            box-shadow: 2px 2px  6px #888;
        }
        [id="sign-language-to-text"],[data-testid="stMarkdownContainer"]{
            color: black !important;
        }
        [data-testid="baseButton-header"]{
            color:black;
        }
        [data-testid="baseButton-secondary"] p{
            color:#e5e5f7;
        }
        .stHeadingContainer h1{
            color: black !important; 
        }
    </style>
"""
st.markdown(page_styles,unsafe_allow_html=True)

# Function to convert live sign to text
def signToText():
    st.title('Sign Language to Text!')

    start_btn_pressed = st.button("Start")
    if start_btn_pressed:
        model_dict_42 = pickle.load(open('./models/model42.p', 'rb'))
        model_42 = model_dict_42['model']

        model_dict_84 = pickle.load(open('./models/model84.p', 'rb'))
        model_84 = model_dict_84['model']

        cap = cv2.VideoCapture(0)

        frame_placeholder = st.empty()
        stop_btn_pressed = st.button("Stop")

        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        hands = mp_hands.Hands(static_image_mode=True,
                               min_detection_confidence=0.3)

      
        labels_dict_42 = {0: '0', 1: '1', 2: '2',3:'3',4: '4', 5: '5', 6: '6',7:'7',8: '8', 9: '9',10:'C',11:'J',12:'L',13:'M',14:'O',15:'U',16:'V'}
        labels_dict_84 = {0: 'A', 1: 'B', 2: 'D',3:'E',4: 'F', 5: 'G', 6: 'H',7:'I',8: 'K', 9: 'N',10:'P', 11: 'R', 12: 'S',13:'T',14: 'W', 15: 'X',16:'Y', 17: 'Z'}

        while cap.isOpened() and not stop_btn_pressed:
            ret, frame = cap.read()

            if not ret:
                st.write("video capturing stopped")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame_placeholder.image(frame_rgb, channels="RGB")

            # For predicting the result
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                x1 = int(min(x_) * frame.shape[1]) - 10
                y1 = int(min(y_) * frame.shape[0]) - 10
                x2 = int(max(x_) * frame.shape[1]) - 10
                y2 = int(max(y_) * frame.shape[0]) - 10
                
                if len(data_aux) == 42:
                    try:
                        prediction = model_42.predict([np.asarray(data_aux)])
                        predicted_character = labels_dict_42[int(prediction[0])]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
                    except Exception as e:
                        print(f"Error predicting: {e}")
                elif len(data_aux) == 84:
                    try:
                        prediction = model_84.predict([np.asarray(data_aux)])
                        predicted_character = labels_dict_84[int(prediction[0])]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3, cv2.LINE_AA)
                    except Exception as e:
                        print(f"Error predicting: {e}")
                else:
                    print("Input size is not 42 or 84 features. Skipping prediction." )
                    
                    
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            # Stop the web cam
            if cv2.waitKey(1) & 0xFF == ord('q') or stop_btn_pressed:
                break

        cap.release()
        cv2.destroyAllWindows()
        print()


# Function to convert text to a sign

def textToSign():
    st.title('Text to Sign Language!')

    def display_images(text):
        img_dir = "./images"

        image_pos = st.empty()
        combined_img = []
        if len(text) == 1:
            if text and text.isalpha():
                img_path = os.path.join(img_dir, f"{text}.png")
                # st.write(img_path)
                img = Image.open(img_path)
                image_pos.image(img, width=500)

                # wait for 2 seconds before displaying the next image
                time.sleep(1)

                # remove the image
                # image_pos.empty()
            elif text and text.isnumeric():
                img_path = os.path.join(img_dir, f"{text}.jpg")
                img = Image.open(img_path)

                image_pos.image(img, width=500)

                # wait for 2 seconds before displaying the next image
                time.sleep(1)

            # remove the image
            # image_pos.empty()
        else:
            # iterate through the text and display sign language images
            for char in text:
                if char.isalpha():
                    img_path = os.path.join(img_dir, f"{char}.png")
                    img = Image.open(img_path)
                    
                    
                    # Combining individual images and displaying the all images at once 
                    combined_img.append(img)
                    # combined_img.append("  ")

                    # image_pos.image(img, width=500)

                    # time.sleep(1)

                    # remove the image
                    # image_pos.empty()
                elif char.isnumeric():
                    img_path = os.path.join(img_dir, f"{char}.jpg")
                    img = Image.open(img_path)

                    # image_pos.image(img, width=500)
                    
                    combined_img.append(img)
                    # combined_img.append("  ")

                    # time.sleep(1)
                elif char == ' ':
                    # display space image for space character
                    img_path = os.path.join(img_dir, "space.png")
                    img = Image.open(img_path)
                    
                    
                    combined_img.append(img)
                    # combined_img.append("  ")
                    
                    # image_pos.image(img, width=500)

                    # time.sleep(1)

                    # remove the image
                    image_pos.empty()

            # wait for 2 seconds before removing the last image
            # time.sleep(2)
            
            image_pos.image(combined_img, width=200)

            # time.sleep()
            # image_pos.empty()x

    text = st.text_input("Enter text:")
    text = text.lower()
    display_images(text)


# Sidebar
st.sidebar.title('Sign Language Recognition App')

# Selection box to select the options
app_mode = st.sidebar.selectbox('Choose the App mode',
['Sign Language to Text', 'Text to sign Language'])

if app_mode == 'Sign Language to Text':
    signToText()
else:
    textToSign()


st.sidebar.caption("Created by : Mohammed Sahil,Mohammed Parvez,Md Sahil, Mohammed Ghouse Pasha")

# cap = cv2.VideoCapture(0)
# st.title("Webcam Testing")

# frame_placeholder = st.empty()
# stop_btn_pressed = st.button("Stop")

# while cap.isOpened() and not stop_btn_pressed:
#     ret,frame = cap.read()

#     if not ret:
#         st.write("video capturing stopped")
#         break

#     frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#     frame_placeholder.image(frame,channels="RGB")

#     if cv2.waitKey(1)  & 0xFF == ord('q') or stop_btn_pressed:
#         break

# cap.release()
# cv2.destroyAllWindows()
