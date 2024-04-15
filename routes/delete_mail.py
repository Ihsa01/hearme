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

delete_mail_bp = Blueprint('delete_mail', __name__)

# Import necessary variables directly from app.py
CLIENT_SECRET_FILE = 'abcc.json'
API_NAME = 'gmail'
API_VERSION = 'v1'
SCOPES = ['https://mail.google.com/', 'https://www.googleapis.com/auth/gmail.readonly']

@delete_mail_bp.route('/delete')




def delete_mail(sender_name):
    """
    Retrieve emails from Gmail inbox and delete the last message from a specific sender
    """
    service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
    results = service.users().messages().list(userId='me', labelIds=['INBOX']).execute()
    messages = results.get('messages', [])

    if not messages:
        return 'No messages found.'

    last_message_id = None
    flag=1
    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id']).execute()
        message_data = msg['payload']['headers']
        for values in message_data:
            name = values['name']
          
            if name == 'From':
                from_name = values['value']
               
                if sender_name in from_name:

                    last_message_id = message['id']
                    flag=0
                    break
        if(flag==0):
            break        
    if last_message_id:
        service.users().messages().delete(userId='me', id=last_message_id).execute()
        print("Last message from {sender_name} deleted successfully!")
        return f"Last message from {sender_name} deleted successfully!"
    else:
        return f"No messages found from {sender_name}."

