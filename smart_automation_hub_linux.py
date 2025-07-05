# ✅ FINAL MERGED CODE: All Linux + Social Media + Voice + Emotion Features
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
import tweepy
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

# --- TTS ---
tts_engine = pyttsx3.init()
def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Listening...")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except:
        return ""

# --- Linux Command Safe Execution ---
def run_command_safely(cmd):
    blocked = ["rm -rf /", "shutdown", "mkfs", "dd if=", ">: /dev/sda"]
    if any(block in cmd.lower() for block in blocked):
        st.error("Unsafe command blocked.")
        speak("Unsafe command")
        return
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        st.code(out.stdout or "[No output]")
    except subprocess.CalledProcessError as e:
        st.error(e.stderr)

# --- Linux Command Interpretation ---
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

# --- Social Media Posting Functions ---
def post_to_linkedin(token, author_urn, post_text):
    try:
        url = "https://api.linkedin.com/v2/ugcPosts"
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Restli-Protocol-Version": "2.0.0",
            "Content-Type": "application/json"
        }
        data = {
            "author": author_urn,
            "lifecycleState": "PUBLISHED",
            "specificContent": {
                "com.linkedin.ugc.ShareContent": {
                    "shareCommentary": {"text": post_text},
                    "shareMediaCategory": "NONE"
                }
            },
            "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"}
        }
        res = requests.post(url, headers=headers, json=data)
        return res.status_code == 201, res.text
    except Exception as e:
        return False, str(e)

def post_to_twitter(api_key, api_secret, access_token, access_secret, tweet):
    try:
        auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_secret)
        api = tweepy.API(auth)
        api.update_status(tweet)
        return True, "Tweet posted successfully!"
    except Exception as e:
        return False, str(e)

def post_to_facebook(page_token, page_id, message):
    try:
        url = f"https://graph.facebook.com/{page_id}/feed"
        payload = {"message": message, "access_token": page_token}
        res = requests.post(url, data=payload)
        return res.ok, res.text
    except Exception as e:
        return False, str(e)

# --- WhatsApp, Email, SMS ---
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

# --- Stock Price Predictor ---
def predict_stock(ticker, days):
    try:
        data = yf.download(ticker, period="5y")
        if data.empty:
            return False, "No data or invalid ticker"
        close_prices = data['Close'].values.reshape(-1,1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_prices)
        x_train, y_train = [], []
        for i in range(60, len(scaled_data)):
            x_train.append(scaled_data[i-60:i, 0])
            y_train.append(scaled_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
        current_input = scaled_data[-60:].reshape(1, 60, 1)
        predicted = []
        for _ in range(days):
            pred = model.predict(current_input)[0,0]
            predicted.append(pred)
            current_input = np.append(current_input[:,1:,:], [[[pred]]], axis=1)
        predicted_prices = scaler.inverse_transform(np.array(predicted).reshape(-1,1))
        return True, predicted_prices
    except Exception as e:
        return False, str(e)

# --- Emotion Detector ---
def detect_emotion():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            cv2.putText(frame, emotion, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        except:
            emotion = ""
        cv2.imshow("Emotion", frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

# --- App Launcher ---
def launch_app(name):
    try:
        subprocess.Popen(name.split())
        speak(f"Launched {name}")
    except:
        speak("Failed to launch app")

# --- Streamlit UI Choices ---
choice = st.sidebar.selectbox("Select Feature", [
    "Run Linux Command",
    "Generate Multiple Linux Commands",
    "Generate HTML Tags",
    "Post to LinkedIn",
    "Post to Twitter",
    "Post to Facebook",
    "Send WhatsApp",
    "Send Email",
    "Send SMS",
    "Make Call",
    "Stock Price Predictor",
    "Emotion Detector",
    "App Launcher"
])

# --- Streamlit UI (continued) ---
if choice == "Post to LinkedIn":
    token = st.text_input("Access Token")
    author_urn = st.text_input("Author URN")
    message = st.text_area("Post Content")
    if st.button("Post"):
        success, msg = post_to_linkedin(token, author_urn, message)
        st.success(msg) if success else st.error(msg)

elif choice == "Post to Twitter":
    api_key = st.text_input("API Key")
    api_secret = st.text_input("API Secret")
    access_token = st.text_input("Access Token")
    access_secret = st.text_input("Access Token Secret")
    tweet = st.text_area("Tweet")
    if st.button("Tweet"):
        success, msg = post_to_twitter(api_key, api_secret, access_token, access_secret, tweet)
        st.success(msg) if success else st.error(msg)

elif choice == "Post to Facebook":
    page_id = st.text_input("Page ID")
    token = st.text_input("Page Access Token")
    msg = st.text_area("Post Message")
    if st.button("Post"):
        success, response = post_to_facebook(token, page_id, msg)
        st.success("Posted!") if success else st.error(response)

elif choice == "Send WhatsApp":
    phone = st.text_input("Phone Number with Country Code")
    msg = st.text_area("Message")
    if st.button("Send WhatsApp"):
        try:
            send_whatsapp(phone, msg)
            st.success("Message Sent")
        except Exception as e:
            st.error(str(e))

elif choice == "Send Email":
    sender = st.text_input("Sender Email")
    password = st.text_input("App Password", type="password")
    recipient = st.text_input("Recipient Email")
    subject = st.text_input("Subject")
    body = st.text_area("Email Body")
    if st.button("Send Email"):
        success, msg = send_email(sender, password, recipient, subject, body)
        st.success(msg) if success else st.error(msg)

elif choice == "Send SMS":
    sid = st.text_input("Twilio SID")
    token = st.text_input("Auth Token", type="password")
    sender_phone = st.text_input("Sender Phone")
    receiver_phone = st.text_input("Receiver Phone")
    text = st.text_area("SMS Content")
    if st.button("Send SMS"):
        success, msg = send_sms(sid, token, sender_phone, receiver_phone, text)
        st.success(msg) if success else st.error(msg)

elif choice == "Make Call":
    sid = st.text_input("Twilio SID")
    token = st.text_input("Auth Token", type="password")
    from_ = st.text_input("From Number")
    to = st.text_input("To Number")
    if st.button("Call"):
        success, msg = make_call(sid, token, from_, to)
        st.success(msg) if success else st.error(msg)

elif choice == "Stock Price Predictor":
    ticker = st.text_input("Enter Ticker Symbol")
    days = st.slider("Prediction Days", 1, 30)
    if st.button("Predict"):
        success, result = predict_stock(ticker, days)
        if success:
            st.line_chart(result)
        else:
            st.error(result)

elif choice == "Emotion Detector":
    if st.button("Start Emotion Detector"):
        detect_emotion()

elif choice == "App Launcher":
    app = st.text_input("Enter app to launch")
    if st.button("Launch App"):
        launch_app(app)

