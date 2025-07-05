import os
import shutil
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

# Initialize TTS
tts_engine = pyttsx3.init()
def speak(text):
    print("Speak:", text)
    tts_engine.say(text)
    tts_engine.runAndWait()

# Voice Command Function
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

def interpret_to_windows_command(prompt):
    system_msg = """
Convert the user's natural language instruction into a single Windows shell command or PowerShell command.
Only return the command.
If not understood, return: echo \"Sorry, I didn’t understand the request.\"
"""
    response = gemini_model.generate_content(f"{system_msg}\nUser: {prompt}\nCommand:")
    return response.text.strip()

def interpret_to_windows_commands(n):
    response = gemini_model.generate_content(f"Generate {n} realistic Windows shell or PowerShell commands without explanation.")
    return response.text.strip().split('\n')

def interpret_to_html_tags(n):
    response = gemini_model.generate_content(f"Generate {n} unique HTML tags enclosed in angle brackets without explanation.")
    return response.text.strip().split('\n')

def run_command_safely(cmd):
    blocked = ["format", "del /f /s /q", "rd /s /q C:", "shutdown /s /t 0"]
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

apps = {
    "notepad": "notepad",
    "firefox": "firefox",
    "vlc": "vlc",
    "calculator": "calc",
    "cmd": "cmd",
    "virtualbox": "VirtualBox",
    "paint": "mspaint",
    "chrome": "chrome",
}

def launch_app(cmd):
    found = False
    for k, v in apps.items():
        if k in cmd.lower():
            os.system(f"start {v}")
            speak(f"Opening {k}")
            found = True
            break
    if not found:
        speak("App not found.")
        st.warning("App not found")

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

def send_whatsapp(phone, msg):
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

# ========== Streamlit App ==========
st.title("AI Integrated Assistant")

if st.sidebar.button("🎙️ Use Voice Command"):
    user_command = listen()
    st.success(f"You said: {user_command}")
    speak(f"You said: {user_command}")
    interpreted = interpret_to_windows_command(user_command)
    st.info(f"Running: {interpreted}")
    run_command_safely(interpreted)

choice = st.sidebar.selectbox("Select Feature", [
    "1: Run Windows Command",
    "2: Multiple Windows Commands",
    "3: HTML Tags",
    "4: Chatbot Gemini",
    "5: App Launcher",
    "6: Scheduler",
    "7: WhatsApp Message",
    "8: Email Sender",
    "9: SMS Sender",
    "10: LinkedIn Poster",
    "11: Twitter Poster",
    "12: Facebook Poster",
])

if choice == "1: Run Windows Command":
    st.subheader("Run Windows Command")
    cmd_input = st.text_input("Enter command:")
    if st.button("Run Command"):
        run_command_safely(cmd_input)

elif choice == "2: Multiple Windows Commands":
    st.subheader("Generate Multiple Commands")
    n = st.number_input("How many?", 1, 20, 5)
    if st.button("Generate"):
        cmds = interpret_to_windows_commands(n)
        for c in cmds:
            st.code(c)

elif choice == "3: HTML Tags":
    st.subheader("Generate HTML Tags")
    n = st.number_input("How many tags?", 1, 50, 10)
    if st.button("Generate Tags"):
        tags = interpret_to_html_tags(n)
        st.write(", ".join(tags))

elif choice == "4: Chatbot Gemini":
    st.subheader("Gemini Chatbot")
    prompt = st.text_area("Talk to Gemini")
    if st.button("Ask"):
        reply = gemini_model.generate_content(prompt)
        st.write(reply.text)

elif choice == "5: App Launcher":
    st.subheader("Launch an App")
    app_cmd = st.text_input("App name:")
    if st.button("Launch"):
        launch_app(app_cmd)

elif choice == "6: Scheduler":
    st.subheader("Schedule a Task")
    date_input = st.date_input("Pick a date")
    time_input = st.time_input("Pick a time")
    task_cmd = st.text_input("Command or launch <app>:")
    if st.button("Schedule"):
        dt = datetime.datetime.combine(date_input, time_input)
        def task_fn():
            if task_cmd.startswith("launch"):
                launch_app(task_cmd.replace("launch", "").strip())
            else:
                run_command_safely(task_cmd)
        tasks.append((dt, task_fn))
        st.success(f"Scheduled for {dt}")

elif choice == "7: WhatsApp Message":
    st.subheader("Send WhatsApp")
    phone = st.text_input("Phone (+countrycode):")
    msg = st.text_area("Message:")
    if st.button("Send WhatsApp"):
        send_whatsapp(phone, msg)
        st.success("Sent!")

elif choice == "8: Email Sender":
    st.subheader("Send Email")
    sender = st.text_input("Your Email:")
    password = st.text_input("App Password:", type="password")
    recipient = st.text_input("To:")
    subject = st.text_input("Subject:")
    body = st.text_area("Body:")
    if st.button("Send Email"):
        success, msg = send_email(sender, password, recipient, subject, body)
        if success:
            st.success(msg)
        else:
            st.error(msg)

elif choice == "9: SMS Sender":
    st.subheader("Send SMS")
    sid = st.text_input("Twilio SID:")
    token = st.text_input("Twilio Token:", type="password")
    sender_phone = st.text_input("Sender:")
    receiver_phone = st.text_input("Receiver:")
    msg = st.text_area("Message:")
    if st.button("Send SMS"):
        success, message = send_sms(sid, token, sender_phone, receiver_phone, msg)
        if success:
            st.success(message)
        else:
            st.error(message)

elif choice == "10: LinkedIn Poster":
    st.subheader("Post to LinkedIn")
    access_token = st.text_input("Access Token:", type="password")
    author_urn = st.text_input("Author URN (e.g. urn:li:person:xxxx):")
    post_text = st.text_area("Post Content:")
    if st.button("Post to LinkedIn"):
        try:
            url = "https://api.linkedin.com/v2/ugcPosts"
            headers = {
                "Authorization": f"Bearer {access_token}",
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
            if res.status_code == 201:
                st.success("Post published to LinkedIn!")
            else:
                st.error(f"Error: {res.text}")
        except Exception as e:
            st.error(str(e))

elif choice == "11: Twitter Poster":
    st.subheader("Post to Twitter (X)")
    api_key = st.text_input("API Key:")
    api_secret = st.text_input("API Secret:", type="password")
    access_token = st.text_input("Access Token:")
    access_secret = st.text_input("Access Secret:", type="password")
    tweet = st.text_area("Tweet Content:")
    if st.button("Post to Twitter"):
        try:
            auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_secret)
            api = tweepy.API(auth)
            api.update_status(tweet)
            st.success("Tweet posted successfully!")
        except Exception as e:
            st.error(str(e))

elif choice == "12: Facebook Poster":
    st.subheader("Post to Facebook Page")
    page_token = st.text_input("Page Access Token:", type="password")
    page_id = st.text_input("Page ID:")
    message = st.text_area("Post Message:")
    if st.button("Post to Facebook"):
        try:
            url = f"https://graph.facebook.com/{page_id}/feed"
            payload = {"message": message, "access_token": page_token}
            res = requests.post(url, data=payload)
            if res.ok:
                st.success("Post published to Facebook!")
            else:
                st.error(res.text)
        except Exception as e:
            st.error(str(e))

