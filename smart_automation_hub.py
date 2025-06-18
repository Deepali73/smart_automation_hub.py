import streamlit as st
import pywhatkit
import smtplib
from twilio.rest import Client
import instaloader
import datetime
import requests
import speech_recognition as sr
import pyttsx3

# ------------------ Text-to-Speech Setup ------------------ #
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# ------------------ Speech Recognition ------------------ #
def listen_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Listening for your command...")
        audio = recognizer.listen(source)
    try:
        command = recognizer.recognize_google(audio)
        st.text(f"You said: {command}")
        return command.lower()
    except sr.UnknownValueError:
        speak("Sorry, I didn't catch that.")
        return ""
    except sr.RequestError:
        speak("Speech service unavailable.")
        return ""

# ------------------ UI Header ------------------ #
st.title("🔧 Smart Automation Hub")
st.markdown("Say or choose an option:")

options = {
    "1": "Send WhatsApp Message",
    "2": "Send Email",
    "3": "Send SMS",
    "4": "Post on Instagram",
    "5": "Download Data",
    "6": "Make a Voice Call",
    "7": "Search Web Data"
}

for k, v in options.items():
    st.markdown(f"{k}. {v}")

# ------------------ User Input ------------------ #
option = st.text_input("Enter your choice (1-7):")

# ------------------ WhatsApp ------------------ #
def send_whatsapp():
    phone = st.text_input("Phone number (with country code):")
    message = st.text_input("Your message:")
    send_time = st.time_input("Time to send:", value=datetime.time(datetime.datetime.now().hour + 1, 0))
    if st.button("📤 Send WhatsApp"):
        try:
            pywhatkit.sendwhatmsg(phone, message, send_time.hour, send_time.minute)
            st.success("WhatsApp message scheduled!")
        except Exception as e:
            st.error(f"Error: {e}")

# ------------------ Email ------------------ #
def send_email():
    sender = st.text_input("Sender Email:")
    password = st.text_input("Password:", type="password")
    receiver = st.text_input("Receiver Email:")
    subject = st.text_input("Subject:")
    body = st.text_area("Email Body:")
    if st.button("📧 Send Email"):
        try:
            msg = f"Subject: {subject}\n\n{body}"
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender, password)
                server.sendmail(sender, receiver, msg)
            st.success("Email sent successfully.")
        except Exception as e:
            st.error(f"Error: {e}")

# ------------------ SMS ------------------ #
def send_sms():
    sid = st.text_input("Twilio SID:")
    token = st.text_input("Auth Token:")
    from_num = st.text_input("From (Twilio number):")
    to_num = st.text_input("To (recipient number):")
    message = st.text_input("SMS Message:")
    if st.button("📱 Send SMS"):
        try:
            client = Client(sid, token)
            client.messages.create(body=message, from_=from_num, to=to_num)
            st.success("SMS sent successfully.")
        except Exception as e:
            st.error(f"Error: {e}")

# ------------------ Instagram Post (Placeholder) ------------------ #
def post_instagram():
    st.info("Due to API restrictions, Instagram posting requires external tools like Selenium or Meta APIs.")
    st.warning("Direct posting via `instaloader` is not supported.")
    # Placeholder for future implementation

# ------------------ Download Data (GET Request Example) ------------------ #
def download_data():
    url = st.text_input("Enter the URL to download data:")
    if st.button("⬇️ Download"):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                st.download_button("Download File", response.content, file_name="downloaded_data.txt")
                st.success("Downloaded successfully!")
            else:
                st.error(f"Failed with status code: {response.status_code}")
        except Exception as e:
            st.error(f"Error: {e}")

# ------------------ Voice Call (Placeholder) ------------------ #
def voice_call():
    st.info("Voice call functionality is dependent on third-party APIs like Twilio voice or Vonage.")
    st.warning("Implementation will require API setup and webhook integration.")

# ------------------ Web Search (Basic Implementation) ------------------ #
def web_search():
    query = st.text_input("Enter your search query:")
    if st.button("🔍 Search"):
        try:
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            st.markdown(f"[Click to view search results]({search_url})", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

# ------------------ Command Execution ------------------ #
def run_selected_option(opt):
    speak(f"You selected option {opt}")
    if opt == "1":
        send_whatsapp()
    elif opt == "2":
        send_email()
    elif opt == "3":
        send_sms()
    elif opt == "4":
        post_instagram()
    elif opt == "5":
        download_data()
    elif opt == "6":
        voice_call()
    elif opt == "7":
        web_search()
    else:
        st.warning("Invalid selection.")

# ------------------ Execute Based on Input ------------------ #
if option:
    run_selected_option(option.strip())

# ------------------ Voice Activation ------------------ #
if st.button("🎤 Speak Command"):
    voice_cmd = listen_command()
    for k, v in options.items():
        if v.lower() in voice_cmd:
            run_selected_option(k)
            break
    else:
        st.warning("Could not understand command. Try again.")
