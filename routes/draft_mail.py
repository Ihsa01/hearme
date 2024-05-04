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
def listenn_and_execute():
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            speak("speak")
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
            speak("Timeout. No speech detected.")
        except sr.UnknownValueError:
            speak("Could not understand audio")
        except sr.RequestError as e:
            speak("Could not request results; {0}".format(e))
        
        attempts += 1
        speak("Please try again.")
    
    speak("Max attempts reached. Exiting...")
    return None



def edit_msg(command, start_word, end_word):
 
    # Find the start and end indices of the substring to replace
    start_index = command.find(start_word)
    end_index = command.find(end_word, start_index + len(start_word))
    
    # If both start and end words are found
    if start_index != -1 and end_index != -1:
        replacement=listen_and_execute()
        edited_command = command[:start_index] + replacement + command[end_index + len(end_word):]
        print( edited_command)
        speak(edited_command)
        return edited_command
    else:
        speak("Start or end word not found in the command.")
        return command

def send_message(service, draft_id, user_id='me'):
    try:
        message = service.users().drafts().send(userId=user_id, body={'id': draft_id}).execute()
        print('Message sent successfully.')
    except Exception as e:
        print(f'An error occurred while sending the message: {e}')


# Define the send_mail_bp blueprint
draft_mail_bp = Blueprint('draft_mail', __name__)

# Import necessary variables directly from app.py
CLIENT_SECRET_FILE = 'abcc.json'
API_NAME = 'gmail'
API_VERSION = 'v1'
SCOPES = ['https://mail.google.com/', 'https://www.googleapis.com/auth/gmail.readonly']

  
  
@draft_mail_bp.route('/draft')

def draft_mail(subject_=None, receiver_email=None):
    speak("Checking drafts")
    print("Checking draft function")
    service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
    try:
        drafts = service.users().drafts().list(userId='me').execute().get('drafts', [])
        if not drafts:
            return "No drafts found."

        if subject_ is not None and receiver_email is not None:
            # Case 1: Both subject and receiver email are provided
            matching_drafts = []
            for draft in drafts:
                draft_id = draft['id']
                draft_info = service.users().drafts().get(userId='me', id=draft_id, format='full').execute()
                headers = draft_info['message']['payload']['headers']

                draft_subject = draft_sender = None
                for header in headers:
                    if header['name'] == 'Subject':
                        draft_subject = header['value']
                    if header['name'] == 'To':
                        draft_sender = header['value']

                if draft_subject is not None and draft_sender is not None:
                    if subject_.lower() in draft_subject.lower():
                        matching_drafts.append((draft_id, draft_subject, draft_sender))
                        message = draft_info.get('message', {})
                        snippet = message.get('snippet', None)
                        if snippet:
                            print("Snippet:", snippet)
                            speak("The content of the draft is")
                            speak(snippet)
                        else:
                            print("Snippet not available.")

                        speak("Here is the draft snippet. Do you want to update it?")
                        confirmation = 'yes'
                        if "yes" in confirmation:
                            # Update draft snippet
                            service.users().drafts().delete(userId='me', id=draft_id).execute()
            
                            speak("Start word")
                            start_word = listen_and_execute()
                            print(start_word)
                            speak("End word")
                            end_word = listen_and_execute()
                            print(end_word) 
                            
                            emailMsg = edit_msg(snippet, start_word, end_word)
                            mimeMessage = MIMEMultipart()
                            mimeMessage['to'] = draft_sender
                            mimeMessage['subject'] = draft_subject
                            mimeMessage.attach(MIMEText(emailMsg, 'plain'))
                            raw_string = base64.urlsafe_b64encode(mimeMessage.as_bytes()).decode()

                            draft = {
                                'message': {
                                    'raw': raw_string
                                }
                            }

                            draft = service.users().drafts().create(userId='me', body=draft).execute()
                            speak("Do you want to send?")
                            flag = listen_and_execute()
                            if 'yes' in flag:
                                send_message(service, draft_id=draft['id'], user_id='me')
                                return "Draft snippet updated and sent successfully"
                            else:
                                return "Draft updated successfully"
                
            if not matching_drafts:
                return "No matching drafts found."

        elif subject_ is not None and receiver_email is None:
            # Case 3: Subject is provided, take the latest draft with the subject
            latest_matching_draft = None
            for draft in drafts:
                draft_id = draft['id']
                draft_info = service.users().drafts().get(userId='me', id=draft_id, format='full').execute()
                headers = draft_info['message']['payload']['headers']
                draft_subject = None
                draft_sender = None
                for header in headers:
                    if header['name'] == 'Subject':
                        draft_subject = header['value']
                        break

                if draft_subject is not None and subject_.lower() in draft_subject.lower():
                    latest_matching_draft = draft_info
                    break
            
            if latest_matching_draft:
                draft_id = latest_matching_draft['id']
                message = latest_matching_draft.get('message', {})
                snippet = message.get('snippet', None)
                if snippet:
                    print("Snippet:", snippet)
                    speak("The content of the draft is")
                    speak(snippet)
                else:
                    print("Snippet not available.")
                
                # Ask if the user wants to update the draft
                speak("Do you want to update this draft?")
                flag = listen_and_execute()
                if 'yes' in flag:
                    # Update draft snippet
                    service.users().drafts().delete(userId='me', id=draft_id).execute()
    
                    speak("Start word")
                    start_word = listen_and_execute()
                    print(start_word)
                    speak("End word")
                    end_word = listen_and_execute()
                    print(end_word) 
                    
                    emailMsg = edit_msg(snippet, start_word, end_word)
                    mimeMessage = MIMEMultipart()
                    mimeMessage['to'] = draft_sender
                    mimeMessage['subject'] = draft_subject
                    mimeMessage.attach(MIMEText(emailMsg, 'plain'))
                    raw_string = base64.urlsafe_b64encode(mimeMessage.as_bytes()).decode()

                    draft = {
                        'message': {
                            'raw': raw_string
                        }
                    }

                    draft = service.users().drafts().create(userId='me', body=draft).execute()
                    speak("Do you want to send?")
                    flag = listen_and_execute()
                    if 'yes' in flag:
                        send_message(service, draft_id=draft['id'], user_id='me')
                        return "Draft updated and sent successfully"
                    else:
                        return "Draft updated successfully"
                else:
                    return "Draft not updated."

            else:
                speak("No draft found with the specified subject")
                return "No draft found with the specified subject."

        elif receiver_email is not None:
            # Case 2: Receiver email is provided, retrieve the most recent draft and send it
            latest_draft = drafts[0]
            latest_draft_id = latest_draft['id']
            latest_draft_info = service.users().drafts().get(userId='me', id=latest_draft_id, format='full').execute()
            headers = latest_draft_info['message']['payload']['headers']
            draft_subject = draft_sender = None

            for header in headers:
                if header['name'] == 'Subject':
                    draft_subject = header['value']
                if header['name'] == 'To':
                    draft_sender = header['value']
            
            if draft_subject is not None and draft_sender is not None:
                message = latest_draft_info.get('message', {})
                snippet = message.get('snippet', None)
                if snippet:
                    print("Snippet:", snippet)
                    speak("The content of the draft is")
                    speak(snippet)
                else:
                    print("Snippet not available.")
                
                speak("Do you want to update this draft?")
                flag = listen_and_execute()
                if 'yes' in flag:
                    # Update draft snippet
                    service.users().drafts().delete(userId='me', id=draft_id).execute()
    
                    speak("Start word")
                    start_word = listen_and_execute()
                    print(start_word)
                    speak("End word")
                    end_word = listen_and_execute()
                    print(end_word) 
                    
                    emailMsg = edit_msg(snippet, start_word, end_word)
                    mimeMessage = MIMEMultipart()
                    mimeMessage['to'] = draft_sender
                    mimeMessage['subject'] = draft_subject
                    mimeMessage.attach(MIMEText(emailMsg, 'plain'))
                    raw_string = base64.urlsafe_b64encode(mimeMessage.as_bytes()).decode()

                    draft = {
                        'message': {
                            'raw': raw_string
                        }
                    }

                    draft = service.users().drafts().create(userId='me', body=draft).execute()
                    speak("Do you want to send?")
                    flag = listen_and_execute()
                    if 'yes' in flag:
                        send_message(service, draft_id=draft['id'], user_id='me')
                        speak("Draft updated and sent successfully")
                        return "Draft updated and sent successfully"
                    else:
                        speak("Draft updated successfully")
                        return "Draft updated successfully"
                else:
                    speak("Draft updated successfully")
                    return "Draft not updated."
                    
            else:
                return "No draft found with the specified subject."
        
        else:
            # Case 4: No parameters provided
            return "Please provide either subject and receiver email, or just the subject or receiver email."

    except Exception as e:
        print("Error:", e)
        return "An error occurred while processing the draft mail request."
