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


def get_message_content(service, user_id, msg_id):
    msg = service.users().messages().get(userId=user_id, id=msg_id, format='full').execute()
    message_content = ""
    if 'parts' in msg['payload']:
        for part in msg['payload']['parts']:
            if part['mimeType'] == 'text/plain':
                message_content += base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
    else:
        message_content += base64.urlsafe_b64decode(msg['payload']['body']['data']).decode('utf-8')
    return message_content

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

search_mail_bp = Blueprint('search_mail', __name__)

# Import necessary variables directly from app.py
CLIENT_SECRET_FILE = 'abcc.json'
API_NAME = 'gmail'
API_VERSION = 'v1'
SCOPES = ['https://mail.google.com/', 'https://www.googleapis.com/auth/gmail.readonly']

@search_mail_bp.route('/search')
def search_mail(sender_name=None, subject=None):
    flag = 1
    service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
    results = service.users().messages().list(userId='me', labelIds=['INBOX']).execute()
    messages = results.get('messages', [])
    if not messages:
        return 'No messages found.'
    
    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id']).execute()
        message_data = msg['payload']['headers']
        for values in message_data:
            name = values['name']
            if name == 'From':
                from_name = values['value']
            elif name == 'Subject':
                message_subject = values['value']
                
        if (sender_name is None or sender_name in from_name) and (subject is None or subject in message_subject.lower()):
            flag = 0
            if 'parts' in msg['payload']:        
                msg_str = base64.urlsafe_b64decode(msg['payload']['parts'][0]['body']['data'].encode('ASCII')).decode('utf-8')
                soup = BeautifulSoup(msg_str, 'html.parser')
                body = soup.get_text()
                print(f"From: {from_name}")
                print(f"Subject: {message_subject}")
                print(f"Body: {body}")
                speak("From")
                speak(from_name)
                speak("on subject")
                speak(message_subject)
                speak("content is")
                speak(body)
            else:
                print("No message body found")
    if flag == 0:
        return "Search successful"   
    else: 
        return "Search unsuccessful"
    

    
            

