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
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders


def create_draft(service, message_body):
    try:
        message = {'message': {'raw': base64.urlsafe_b64encode(message_body.encode()).decode()}}
        draft = service.users().drafts().create(userId='me', body=message).execute()
        draft_id = draft['id']
        print(f"Draft created successfully. Draft ID: {draft_id}")
        return draft_id
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen_and_execute(max_attempts=10):
    recognizer = sr.Recognizer()
    attempts = 0

    while attempts < max_attempts:
        try:
            with sr.Microphone() as source:
                print("Listening...")
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
                print("Recognizing...")
                command = recognizer.recognize_google(audio).lower()
                if 'draught' in command:
                    command = command.replace("draught", "draft")
                elif 'draughts' in command:
                    command = command.replace("draughts", "drafts")
                print("You said:", command)
                speak(command)
                return command
        except sr.WaitTimeoutError:
            print("Timeout. No speech detected.")
        except sr.UnknownValueError:
            speak("Could not understand audio")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        
        attempts += 1
        speak("Please try again.")
    
    speak("Max attempts reached. Exiting...")
    return None


def listenandexecute(max_attempts=10):
    recognizer = sr.Recognizer()
    attempts = 0

    while attempts < max_attempts:
        try:
            with sr.Microphone() as source:
                print("Listening...")
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
                print("Recognizing...")
                command = recognizer.recognize_google(audio).lower()
                if "dot" in command:
                    command = command.replace(" dot ", ".")
                if 'draught' in command:
                    command = command.replace("draught", "draft")
                elif 'draughts' in command:
                    command = command.replace("draughts", "drafts")
                print("You said:", command)
                speak(command)
                return command
        except sr.WaitTimeoutError:
            print("Timeout. No speech detected.")
        except sr.UnknownValueError:
            speak("Could not understand audio")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        
        attempts += 1
        speak("Please try again.")
    
    speak("Max attempts reached. Exiting...")
    return None


            
def listen_fast():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=20)
            print("Recognizing...")
            command = recognizer.recognize_google(audio).lower()
            speak("You said:")
            speak(command)
            speak("do you wanna edit at any parts if yes or no ") 
            flag= listen_and_execute()
            speak("you said"+flag)
            if "yes" in flag:
                speak("start word")  
                start_word  = listen_and_execute()
                speak("end word")
                end_word  = listen_and_execute()
    
                command=edit_msg(command,start_word,end_word)
                return command
            return command

            
    except sr.WaitTimeoutError:
        print("Timeout. No speech detected.")
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))


def edit_msg(command, start_word, end_word):
 
    # Find the start and end indices of the substring to replace
    start_index = command.find(start_word)
    end_index = command.find(end_word, start_index + len(start_word))
    
    # If both start and end words are found
    if start_index != -1 and end_index != -1:
        replacement=listen_and_execute()
        edited_command = command[:start_index] + replacement + command[end_index + len(end_word):]
        print( edited_command)
        return edited_command
    else:
        speak("Start or end word not found in the command.")
        return command
        
def listen_slow():
    f=1
    command=listen_and_execute()
    while(f):
       
        speak(command)
        speak("do you wanna edit at any parts if yes or no ") 
        flag= listen_and_execute()
        speak("you said"+flag)
        if "yes" in flag:
            speak("start word")  
            start_word  = listen_and_execute()
            speak("end word")
            end_word  = listen_and_execute()
    
            command=edit_msg(command,start_word,end_word)
        speak("do you want to add anything")
        flag=listen_and_execute()
        if("no" in flag):
            return command
        command=command+listen_and_execute()
 

def dictate_email_body():
    while(1):
        speak("How do you want to dictate the body of the email slow or fast")
        speed = listen_and_execute()
        if "fast" in speed:
            speak("dictate body")
            return listen_fast()
        
        elif "slow" in speed:
            return listen_slow()
        else:
            print("Invalid input. Please enter 'slow' or 'fast'.")

# Define the send_mail_bp blueprint
send_mail_bp = Blueprint('send_mail', __name__)

# Import necessary variables directly from app.py
CLIENT_SECRET_FILE = 'abcc.json'
API_NAME = 'gmail'
API_VERSION = 'v1'
SCOPES = ['https://mail.google.com/', 'https://www.googleapis.com/auth/gmail.readonly']

  
  
@send_mail_bp.route('/send')
def send_mail(reciever_email,subject_):
    """
    Send email using Gmail API
    """
    service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

    emailMsg = dictate_email_body()

    attachments = []
    while True:
        speak("Do you want to attach another file?")
        a=listenandexecute()
        if "yes" in a:
            speak("say filename")
            filename = listenandexecute()
            file_path = os.path.join(r'C:\Users\logOn\OneDrive\Desktop', filename)
            print(file_path)
            if os.path.exists(file_path):  # Check if the file exists
                print("yes")
                attachments.append(file_path)
            else:
                print(f"File '{filename}' not found. Please enter a valid filename.")
        else:
            break

        if not attachments:
            return "No attachments provided."

       

    mimeMessage = MIMEMultipart()
    mimeMessage['to'] = reciever_email
    mimeMessage['subject'] =  subject_
    mimeMessage.attach(MIMEText(emailMsg, 'plain'))
    for attachment in attachments:
        part = MIMEBase('application', 'octet-stream')
        with open(attachment, 'rb') as file:
            part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename= {os.path.basename(attachment)}')
        mimeMessage.attach(part) 
    raw_string = base64.urlsafe_b64encode(mimeMessage.as_bytes()).decode()
    raw_message = mimeMessage.as_string()
    # Ask for confirmation before sending
    speak("Do you want to send this email?")
    confirmation = listenandexecute()
    speak(confirmation)
    if "yes" in confirmation:
        message = service.users().messages().send(userId='me', body={'raw': raw_string}).execute()
        print( 'Email sent successfully')
        speak( 'Email sent successfully')
        return ("Email sent successfully")
    else:
        draft_id = create_draft(service, raw_message)
        if draft_id:
            speak( 'draft saved succesfully')
            return ("draft saved successfully")
        else:
            speak( 'draft saved succesfully')
            return ('draft saved succesfully')

   
