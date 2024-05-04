import re
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
            pass
            return command
    except sr.WaitTimeoutError:
        print("Timeout. No speech detected.")
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

read_mail_bp = Blueprint('read_mail', __name__)

# Import necessary variables directly from app.py
CLIENT_SECRET_FILE = 'abcc.json'
API_NAME = 'gmail'
API_VERSION = 'v1'
SCOPES = ['https://mail.google.com/', 'https://www.googleapis.com/auth/gmail.readonly']

@read_mail_bp.route('/read')

def extract_email_from_header(header):
    match = re.search(r'<([^>]+)>', header)
    if match:
        return match.group(1)
    return None

def read_mail(sender_email):
    """
    Retrieve the last email from a specific sender in the Gmail inbox and read out its content using speak()
    """
    print("Reading mails")
    
    service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
    results = service.users().messages().list(userId='me', labelIds=['INBOX']).execute()
    messages = results.get('messages', [])

    last_email_from_sender = None

    if not messages:
        speak('No messages found.')
        return 'No messages found.'
    else:
        speak("Reading latest mail")
        print("Messages:")
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            message_data = msg['payload']['headers']
            from_name = None
            subject = None
            for values in message_data:
                name = values['name']
                if name == 'From':
                    from_name = values['value']
                if name == 'Subject':
                    subject = values['value']
            if extract_email_from_header(from_name) == sender_email:
                last_email_from_sender = msg
                break
            
        if last_email_from_sender:
            msg_str = base64.urlsafe_b64decode(last_email_from_sender['payload']['parts'][0]['body']['data'].encode('ASCII')).decode('utf-8')
            soup = BeautifulSoup(msg_str, 'html.parser')
            body = soup.get_text()
            print("From:" + from_name)
            print("Subject: " + subject)
            print(body)
            speak(f"From: {from_name}")
            speak("On subject")
            speak(subject)
            speak("Content of the mail is")
            speak(body)
            return("read email successfully")
        else:
            speak(f"No email found from {sender_email}.")
            return("No email found")

