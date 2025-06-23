import os
import subprocess
import threading
import time
import datetime
import requests
import smtplib
import streamlit as st
import speech_recognition as sr
import pyttsx3
from twilio.rest import Client
import pyautogui
import google.generativeai as genai
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import cv2
import urllib.parse
import traceback
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import mediapipe as mp
from deepface import DeepFace

# --- Gemini Setup ---
GEMINI_API_KEY = "AIzaSyB0iLKcdt1aB2blR3CGQibRbDLLbnci8ro"
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("models/gemini-2.5-flash-preview-05-20")

# Initialize TTS engine
tts_engine = pyttsx3.init()
def speak(text):
    print("Speak:", text)
    tts_engine.say(text)
    tts_engine.runAndWait()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as src:
        speak("Listening...")
        r.adjust_for_ambient_noise(src)
        audio = r.listen(src)
    try:
        return r.recognize_google(audio).lower()
    except:
        return ""

def interpret_to_linux_command(prompt):
    system_msg = """
Convert the user's natural language instruction into a single Linux shell command.
Only return the command.
If not understood, return: echo "Sorry, I didn’t understand the request."
"""
    response = gemini_model.generate_content(f"{system_msg}\nUser: {prompt}\nCommand:")
    return response.text.strip()

def interpret_to_linux_commands(n):
    response = gemini_model.generate_content(f"Generate {n} realistic Linux shell commands without explanation.")
    return response.text.strip().split('\n')

def interpret_to_html_tags(n):
    response = gemini_model.generate_content(f"Generate {n} unique HTML tags enclosed in angle brackets without explanation.")
    return response.text.strip().split('\n')

def run_command_safely(cmd):
    # Basic safety checks for Linux dangerous commands
    blocked = ["rm -rf /", "shutdown now", "mkfs", ":(){ :|:& };:", "dd if=", ">: /dev/sda"]
    if any(block in cmd.lower() for block in blocked):
        st.error("Unsafe command blocked.")
        speak("Unsafe command detected")
        return
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        st.code(out.stdout or "[No output]")
        speak("Command executed successfully")
    except subprocess.CalledProcessError as e:
        st.error(e.stderr)
        speak("Execution error occurred")

# Linux App Launcher
apps = {
    "gedit": "gedit",
    "firefox": "firefox",
    "vlc": "vlc",
    "calculator": "gnome-calculator",
    "terminal": "gnome-terminal",
    "virtualbox": "virtualbox",
    "gimp": "gimp",
    "chrome": "google-chrome",
}

def launch_app(cmd):
    found = False
    for k, v in apps.items():
        if k in cmd.lower():
            try:
                subprocess.Popen([v])
                speak(f"Opening {k}")
                found = True
                break
            except Exception as e:
                st.error(f"Failed to open {k}: {e}")
                speak(f"Failed to open {k}")
                found = True
                break
    if not found:
        speak("App not found.")
        st.warning("App not found")

# Scheduler
tasks = []

def schedule_loop():
    while True:
        now = datetime.datetime.now()
        for t, fn in tasks.copy():
            if now >= t:
                threading.Thread(target=fn, daemon=True).start()
                tasks.remove((t, fn))
        time.sleep(1)

threading.Thread(target=schedule_loop, daemon=True).start()

def send_whatsapp(phone,msg):
    import pywhatkit
    pywhatkit.sendwhatmsg_instantly(phone, msg, wait_time=10, tab_close=True)
    time.sleep(20)
    pyautogui.press("enter")

def send_email(sender, password, to, subject, body):
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as srv:
            srv.starttls()
            srv.login(sender, password)
            srv.sendmail(sender, to, f"Subject:{subject}\n\n{body}")
        return True, "Email sent"
    except Exception as e:
        return False, str(e)

def send_sms(sid, token, sender, receiver, msg):
    try:
        Client(sid, token).messages.create(body=msg, from_=sender, to=receiver)
        return True, "SMS sent"
    except Exception as e:
        return False, str(e)

def make_call(sid, token, from_, to):
    try:
        call = Client(sid, token).calls.create(
            url='http://demo.twilio.com/docs/voice.xml', from_=from_, to=to)
        return True, f"Call initiated: {call.sid}"
    except Exception as e:
        return False, str(e)

def download_data_from_url(url):
    try:
        r = requests.get(url)
        if r.ok:
            return True, r.content
        return False, r.status_code
    except Exception as e:
        return False, str(e)

# ================== EMOTION DETECTOR with YouTube Playback ==================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

emotion_queries = {
    "happy": "hindi happy dance songs",
    "sad": "hindi sad songs playlist",
    "angry": "hindi angry pump up songs"
}

track_order = list(emotion_queries.keys())
track_index = 0
last_emotion = ""
last_side = ""
last_play_time = 0
current_driver = None

def play_on_youtube(search_query):
    global current_driver

    if current_driver:
        try:
            current_driver.quit()
            speak("Closed previous video.")
        except Exception as e:
            speak("Error closing previous video.")

    try:
        speak(f"Searching YouTube for: {search_query}")
        query = urllib.parse.quote(search_query)
        search_url = f"https://www.youtube.com/results?search_query={query}"

        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-extensions")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get(search_url)

        time.sleep(3)
        first_video = driver.find_element(By.ID, "video-title")
        first_video.click()
        current_driver = driver
        speak("Playing video now.")

    except Exception:
        speak("YouTube playback error occurred.")
        traceback.print_exc()

def detect_emotion_streamlit():
    global track_index, last_emotion, last_side, last_play_time

    cap = cv2.VideoCapture(0)
    time.sleep(2)
    stframe = st.empty()

    # Preload DeepFace model to avoid delay
    _ = DeepFace.analyze(
        img_path=np.zeros((224, 224, 3), dtype=np.uint8),
        actions=['emotion'],
        enforce_detection=False,
        detector_backend='mediapipe'
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            speak("Camera not found or disconnected.")
            break

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        try:
            emotion_result = DeepFace.analyze(
                rgb,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='mediapipe'
            )
            emotion = emotion_result[0]['dominant_emotion']
            cv2.putText(frame, f"Emotion: {emotion}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            current_time = time.time()
            if emotion in emotion_queries and emotion != last_emotion and (current_time - last_play_time) > 10:
                speak(f"Detected emotion is {emotion}. Playing related videos.")
                play_on_youtube(emotion_queries[emotion])
                last_emotion = emotion
                track_index = track_order.index(emotion)
                last_play_time = current_time

        except Exception:
            speak("Emotion detection error occurred.")
            traceback.print_exc()

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_eye = landmarks[33]
            right_eye = landmarks[263]
            face_center_x = (left_eye.x + right_eye.x) / 2

            if face_center_x < 0.35 and last_side != "left":
                track_index = (track_index + 1) % len(track_order)
                emotion = track_order[track_index]
                play_on_youtube(emotion_queries[emotion])
                last_emotion = emotion
                last_play_time = time.time()
                last_side = "left"

            elif face_center_x > 0.65 and last_side != "right":
                track_index = (track_index - 1) % len(track_order)
                emotion = track_order[track_index]
                play_on_youtube(emotion_queries[emotion])
                last_emotion = emotion
                last_play_time = time.time()
                last_side = "right"

            elif 0.35 <= face_center_x <= 0.65:
                last_side = ""

        cv2.putText(frame, "Language: Hindi", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if st.button("Stop Emotion Detector"):
            speak("Stopping emotion detector.")
            break

    cap.release()
    cv2.destroyAllWindows()
    if current_driver:
        try:
            current_driver.quit()
        except:
            pass

# ============ Streamlit UI ============

st.title("Advanced AI Integrated App (Linux)")

choice = st.sidebar.selectbox("Select Feature", [
    "1: Run Linux Command",
    "2: Multiple Linux Commands",
    "3: HTML Tags",
    "4: Chatbot Gemini",
    "5: App Launcher",
    "6: Scheduler",
    "7: WhatsApp Message",
    "8: Email Sender",
    "9: SMS Sender",
    "10: Emotion Detector",
    "11: Stock Price Predictor",
])

if choice == "1: Run Linux Command":
    st.header("Run a single Linux command")
    cmd_input = st.text_input("Enter your command:")
    if st.button("Run Command"):
        run_command_safely(cmd_input)

elif choice == "2: Multiple Linux Commands":
    st.header("Generate multiple Linux commands")
    n = st.number_input("How many commands?", min_value=1, max_value=20, value=5)
    if st.button("Generate Commands"):
        cmds = interpret_to_linux_commands(n)
        for c in cmds:
            st.code(c)

elif choice == "3: HTML Tags":
    st.header("Generate HTML Tags")
    n = st.number_input("How many HTML tags?", min_value=1, max_value=50, value=10)
    if st.button("Generate HTML Tags"):
        tags = interpret_to_html_tags(n)
        st.write(", ".join(tags))

elif choice == "4: Chatbot Gemini":
    st.header("Chatbot Gemini AI")
    user_prompt = st.text_area("Talk to Gemini:")
    if st.button("Send"):
        response = gemini_model.generate_content(user_prompt)
        st.write(response.text)

elif choice == "5: App Launcher":
    st.header("Launch Linux Apps")
    app_cmd = st.text_input("App name or command:")
    if st.button("Launch"):
        launch_app(app_cmd)

elif choice == "6: Scheduler":
    st.header("Schedule a Task")
    date_input = st.date_input("Select date")
    time_input = st.time_input("Select time")
    task_cmd = st.text_input("Task command or app to launch")

    if st.button("Schedule Task"):
        dt = datetime.datetime.combine(date_input, time_input)
        if dt <= datetime.datetime.now():
            st.error("Cannot schedule task in the past.")
        else:
            def task_fn():
                if task_cmd.startswith("launch"):
                    launch_app(task_cmd.replace("launch","").strip())
                else:
                    run_command_safely(task_cmd)
            tasks.append((dt, task_fn))
            speak(f"Task scheduled for {dt}")
            st.success(f"Task scheduled for {dt}")

elif choice == "7: WhatsApp Message":
    st.header("Send WhatsApp Message")
    phone = st.text_input("Enter phone number (with country code):")
    msg = st.text_area("Message:")
    if st.button("Send WhatsApp"):
        try:
            send_whatsapp(phone, msg)
            st.success("WhatsApp message sent!")
        except Exception as e:
            st.error(str(e))

elif choice == "8: Email Sender":
    st.header("Send Email")
    sender = st.text_input("Your Email:")
    password = st.text_input("App Password:", type="password")
    recipient = st.text_input("Recipient Email:")
    subject = st.text_input("Subject:")
    body = st.text_area("Body:")
    if st.button("Send Email"):
        success, msg = send_email(sender, password, recipient, subject, body)
        if success:
            st.success(msg)
        else:
            st.error(msg)

elif choice == "9: SMS Sender":
    st.header("Send SMS via Twilio")
    sid = st.text_input("Twilio SID:")
    token = st.text_input("Twilio Token:", type="password")
    sender_phone = st.text_input("Sender Phone:")
    receiver_phone = st.text_input("Receiver Phone:")
    sms_msg = st.text_area("Message:")
    if st.button("Send SMS"):
        success, msg = send_sms(sid, token, sender_phone, receiver_phone, sms_msg)
        if success:
            st.success(msg)
        else:
            st.error(msg)

elif choice == "10: Emotion Detector":
    st.header("Emotion Detector with YouTube Playback")
    st.write("Using webcam to detect your emotion and play Hindi songs accordingly.")

    if st.button("Start Emotion Detector"):
        detect_emotion_streamlit()

elif choice == "11: Stock Price Predictor":
    st.header("Stock Price Predictor")
    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL):")
    days = st.number_input("Days to Predict", min_value=1, max_value=60, value=30)
    if st.button("Predict"):
        try:
            data = yf.download(ticker, period="5y")
            if data.empty:
                st.error("Invalid ticker or no data.")
            else:
                close_prices = data['Close'].values.reshape(-1,1)
                scaler = MinMaxScaler(feature_range=(0,1))
                scaled_data = scaler.fit_transform(close_prices)
                x_train = []
                y_train = []
                for i in range(60, len(scaled_data)):
                    x_train.append(scaled_data[i-60:i, 0])
                    y_train.append(scaled_data[i, 0])
                x_train, y_train = np.array(x_train), np.array(y_train)
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

                model = Sequential()
                model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
                model.add(LSTM(units=50))
                model.add(Dense(1))
                model.compile(loss='mean_squared_error', optimizer='adam')
                model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)

                inputs = scaled_data[-60:]
                predicted = []
                current_input = inputs.reshape(1, 60, 1)
                for _ in range(days):
                    pred = model.predict(current_input)[0,0]
                    predicted.append(pred)
                    current_input = np.append(current_input[:,1:,:], [[[pred]]], axis=1)

                predicted_prices = scaler.inverse_transform(np.array(predicted).reshape(-1,1))

                plt.figure(figsize=(10,5))
                plt.plot(data.index[-days:], predicted_prices, label="Predicted Prices")
                plt.legend()
                st.pyplot(plt)
        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
    st.write("Select a feature from the sidebar.")
