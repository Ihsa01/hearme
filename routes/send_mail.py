from flask import Flask, Blueprint, request, current_app
from Google import Create_Service
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from bs4 import BeautifulSoup
from playsound import playsound
import pyttsx3
import os
import speech_recognition as sr






def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen_and_execute():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
            print("Recognizing...")
            command = recognizer.recognize_google(audio).lower()
            speak("You said:")
            speak(command)
            pass
            return command
    except sr.WaitTimeoutError:
        print("Timeout. No speech detected.")
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

# Define the send_mail_bp blueprint
send_mail_bp = Blueprint('send_mail', __name__)

# Import necessary variables directly from app.py
CLIENT_SECRET_FILE = 'abcc.json'
API_NAME = 'gmail'
API_VERSION = 'v1'
SCOPES = ['https://mail.google.com/', 'https://www.googleapis.com/auth/gmail.readonly']

@send_mail_bp.route('/send')
def send_mail(slots):
    from app import Users, db
    
    print("sendmailacti")
    # print(slots)
    # reciever_name = slots.get('B-reciever')
    with current_app.app_context():
    # Access the db to reflect the tables
        db.reflect()
        reciever = Users.query.filter_by(username="asna").first()
        if reciever:
            reciever_email = reciever.email
        else:
            return "reciever not found in database"

        print(reciever_email)


        # Extract subject from the provided slots
        # subject = slots.get('B-subject')
        """
        Send email using Gmail API
        """
        service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
    
        emailMsg = "Hiiiii"

        mimeMessage = MIMEMultipart()
        mimeMessage['to'] = reciever_email
        mimeMessage['subject'] = "demo"
        mimeMessage.attach(MIMEText(emailMsg, 'plain'))
        raw_string = base64.urlsafe_b64encode(mimeMessage.as_bytes()).decode()

        message = service.users().messages().send(userId='me', body={'raw': raw_string}).execute()
    
    
    return redirect(url_for('homee'))
