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
   
    service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
    results = service.users().messages().list(userId='me', labelIds=['INBOX'], q='is:unread').execute()
    messages = results.get('messages', [])

    if not messages:
        return 'No unread messages found.'

    unread_message_id = None
    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id']).execute()
        message_data = msg['payload']['headers']
        for values in message_data:
            name = values['name']
          
            if name == 'From':
                from_name = values['value']
               
                if sender_name in from_name:
                    unread_message_id = message['id']
                    break

    if unread_message_id:
        msg = service.users().messages().get(userId='me', id=unread_message_id).execute()
        message_subject = None
        for header in msg['payload']['headers']:
            if header['name'] == 'Subject':
                message_subject = header['value']
                break
        
        speak("This mail is unread, are you sure you want to delete this")
        confirmation = listen_and_execute()
        if 'yes' in confirmation:
            service.users().messages().delete(userId='me', id=unread_message_id).execute()
            print("Unread message from " + sender_name + " deleted successfully!")
            speak("deleted successfully")
            speak("message deleted successfully")
            return "Deleted"
        else:
            print("Message not deleted.")
            speak("Message not deleted")
            return "Not deleted"
    else:
        speak("No messages found from the user")
        return "Not found. Try again."
    

