from flask import Flask,redirect, url_for, request
from flask_sqlalchemy import SQLAlchemy
from Google import Create_Service
from flask_migrate import Migrate
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
from email.mime.base import MIMEBase            
from email import encoders
import numpy as np
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import pickle as pk
from tensorflow.keras.layers import Dropout, Dense
from routes.send_mail import send_mail



#speech to text
    
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()



#texttospeech
    
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


BERT_MODEL = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
tf_bert_model = TFBertModel.from_pretrained(BERT_MODEL)

class JointIntentAndSlotFillingModel(tf.keras.Model):
    def __init__(self, total_intent_no=None, total_slot_no=None,
                 model_name=BERT_MODEL, dropout_prob=0.1):
        super().__init__()
        self.bert = TFBertModel.from_pretrained(model_name)
        self.dropout = Dropout(dropout_prob)
        self.intent_classifier = Dense(total_intent_no, activation='softmax')
        self.slot_classifier = Dense(total_slot_no, activation='softmax')

    def call(self, inputs, training=None, **kwargs):
        # Retrieve input_ids and attention_masks from the inputs dictionary
        input_ids = inputs['input_ids']
        attention_masks = inputs['attention_masks']

        # Process input through BERT model
        bert_output = self.bert(input_ids, attention_mask=attention_masks, training=training)

        # Use the pooled_output for intent classification
        pooled_output = self.dropout(bert_output.pooler_output, training=training)
        intent_predicted = self.intent_classifier(pooled_output)

        # Use the sequence_output for slot filling
        sequence_output = self.dropout(bert_output.last_hidden_state, training=training)
        slots_predicted = self.slot_classifier(sequence_output)

        return slots_predicted, intent_predicted

#Load the trained model checkpoint
def load_model_checkpoint():
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'joint_model')
    # Get the latest checkpoint file in the directory
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    
    # Check if the latest_checkpoint is not None before proceeding to load the model
    if latest_checkpoint:
        # Load the model using the latest checkpoint
        model = JointIntentAndSlotFillingModel(total_intent_no=3, total_slot_no=9, dropout_prob=0.1)
        status=model.load_weights(latest_checkpoint)
        status.expect_partial()
        return model
    else:
        print("No checkpoint files found in the directory:", checkpoint_dir)

# # Define the function to process slot predictions
def process_slots_prediction(slots_predicted):
    # Define a dictionary to map slot indices to slot labels
    slot_labels = {
        0: "UNK",  # Padding token
        1: "O",    # Outside of a named entity
        2: "B-sender",  # Beginning of an action
        3: "I-sender",
        4: "I-subject",
        5: "B-reciever",
        6: "B-subject",
        7: "I-reciever" # Inside of an action
        # Add more slot labels as needed based on your model's output
    }

    # Initialize an empty list to store the processed slots
    processed_slots = []
    
    # Iterate over each predicted slot
    for slot_index in slots_predicted.flatten():  # Flatten the NumPy array
        # Map the slot index to the corresponding slot label
        slot_index=int(slot_index)
        slot_label = slot_labels[slot_index]
        
        # Add the slot label to the list of processed slots
        processed_slots.append(slot_label)
    
    return processed_slots

# # Load the model and label encoder
model = load_model_checkpoint()
with open('intent_label_encoder.pkl', 'rb') as le_file:
    label_encoder = pk.load(le_file)
with open('seq_out_index_word.pkl', 'rb') as le_file:
    index_to_word = pk.load(le_file)

# Define the function to predict intents and slots
def predict_intent_slots(text):
    tokenized_text = tokenizer.encode(text, add_special_tokens=True, max_length=128, truncation=True)
    input_ids = np.array([tokenized_text])
    attention_masks = np.ones_like([tokenized_text])

    bert_output = tf_bert_model(input_ids, attention_mask=attention_masks)
    input_data = {"input_ids": np.array([tokenized_text]), "attention_masks": np.ones_like([tokenized_text])}
    slots_predicted, intent_predicted = model.predict(input_data)

    intent = label_encoder.inverse_transform([np.argmax(intent_predicted)])
    slots = np.argmax(slots_predicted, axis=-1)
    print("INTENT: " + intent[0])

    merged_tokens = []
    merged_labels = []

    current_token = ""
    current_label = ""
    token_label_dict = {}

    for token_id, label_id in zip(tokenized_text, slots[0]):
        token = tokenizer.decode([token_id])
        label = index_to_word[label_id]
        if token.startswith("##"):  # Handle subwords
            current_token += token[2:]
        else:
            if current_token and current_label:  # If there's a previous token and label
                if current_label != "O" and current_label != "PAD":  # Exclude "O" and "PAD" labels
                    token_label_dict[current_label] = current_token
                merged_tokens.append(current_token)
                merged_labels.append(current_label)
            current_token = token
            current_label = label

    # Add the last token and label
    if current_token and current_label:
        if current_label != "O" and current_label != "PAD":  # Exclude "O" and "PAD" labels
            token_label_dict[current_label] = current_token
        merged_tokens.append(current_token)
        merged_labels.append(current_label)

    

    return intent[0], token_label_dict




#gmailapi

CLIENT_SECRET_FILE = 'abcc.json'
API_NAME = 'gmail'
API_VERSION = 'v1'
SCOPES = ['https://mail.google.com/', 'https://www.googleapis.com/auth/gmail.readonly']



#database

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///my_database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


migrate = Migrate(app, db)


class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    
    pass







def listen_fast():
    recognizer = sr.Recognizer()

    try:
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=30)
            print("Recognizing...")
            command = recognizer.recognize_google(audio).lower()
            speak("You said:")
            speak(command)
            speak("do you wanna edit at any parts if yes or no ") 
            flag= listen_and_execute()
            if(flag!="no"):
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
        speak("do you want to add anything")
        flag=listen_and_execute()
        if(flag=="no"):
            return command
        command=command+listen_and_execute()
 

def dictate_email_body():
    while(1):
        speak("How do you want to dictate the body of the email slow or fast")
        speed = listen_and_execute()
        if speed == "fast":
            return listen_fast()
        
        elif speed == "slow":
            return listen_slow()
        else:
            print("Invalid input. Please enter 'slow' or 'fast'.")

        


def read_emails():
    """
    Retrieve emails from Gmail inbox
    """
    speak("reading mails")
    service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
    results = service.users().messages().list(userId='me', labelIds=['INBOX']).execute()
    messages = results.get('messages', [])

    if not messages:
        return 'No messages found.'
    else:
        print("Messages:")
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            message_data = msg['payload']['headers']
            for values in message_data:
                name = values['name']
               
                if name == 'From':
                    from_name = values['value']
                    speak(from_name)
                if name == 'Subject':
                    subject = values['value']
            if 'parts' in msg['payload']:        
                msg_str = base64.urlsafe_b64decode(msg['payload']['parts'][0]['body']['data'].encode('ASCII')).decode('utf-8')
                soup = BeautifulSoup(msg_str, 'html.parser')
                body = soup.get_text()
                print(f"From: {from_name}")
                print(f"Subject: {subject}")
            else:
                print("noooooo") 



def delete_last_message_from_sender(sender_name):
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
                print(from_name)
                if sender_name in from_name:

                    last_message_id = message['id']
                    flag=0
                    break
        if(flag==0):
            break        
    if last_message_id:
        service.users().messages().delete(userId='me', id=last_message_id).execute()
        return f"Last message from {sender_name} deleted successfully!"
    else:
        return f"No messages found from {sender_name}."



def search_email(sender_name):
    
    service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
    results = service.users().messages().list(userId='me', labelIds=['INBOX']).execute()
    messages = results.get('messages', [])

    if not messages:
        return 'No messages found.'
    flag=1
    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id']).execute()
        message_data = msg['payload']['headers']
        for values in message_data:
            name = values['name']
          
            if name == 'From':
                from_name = values['value']
              
                if sender_name in from_name:
                    flag=0
                    break
        if(flag==0):
            break    
    if(flag==0):
        speak("yes there is") 
    else:
        speak("no message")   




def send_email_att():

    """fl= listen_and_executee()
    print(fl)"""
    filename=listen_and_execute()
    file_path = os.path.join(r'C:\Users\AJEES\Documents', filename)
    if os.path.exists(file_path):  # Check if the file exists
            print("yes")
    else:
            print(f"File  not found. Please enter a valid filename.")

   
   
    service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

    # Your email content
    emailMsg = 'Here is your attachment.'
    mimeMessage = MIMEMultipart()
    mimeMessage['to'] = 'asna030502@gmail.com'
    mimeMessage['subject'] = 'Email with Attachment'
    mimeMessage.attach(MIMEText(emailMsg, 'plain'))
    with open(file_path, 'rb') as file:
        attachment = MIMEBase('application', 'octet-stream')
        attachment.set_payload(file.read())

    # Encode file in base64
    encoders.encode_base64(attachment)

    # Add headers to attachment
    attachment.add_header(
        'Content-Disposition',
        f'attachment; filename= {os.path.basename(file_path)}'
    )

    # Attach the attachment to the email message
    mimeMessage.attach(attachment)

    # Convert message to string and encode as base64
    raw_string = base64.urlsafe_b64encode(mimeMessage.as_bytes()).decode()

    # Send the email
    message = service.users().messages().send(userId='me', body={'raw': raw_string}).execute()

    return 'Email sent successfully!'





def create_draft_email():
    """
    Create a draft email using Gmail API
    """
    service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)

    emailMsg = 'This is a draft email.'
    mimeMessage = MIMEMultipart()
    mimeMessage['to'] = 'aanittantony@gmail.com'
    mimeMessage['subject'] = 'Draft Email'
    mimeMessage.attach(MIMEText(emailMsg, 'plain'))
    raw_string = base64.urlsafe_b64encode(mimeMessage.as_bytes()).decode()

    draft = {
        'message': {
            'raw': raw_string
        }
    }

    draft = service.users().drafts().create(userId='me', body=draft).execute()
    return 'Draft email created successfully!' 

from routes.send_mail import send_mail_bp
app.register_blueprint(send_mail_bp)

@app.route('/')
def homee():
    print("route workinggggg")
    user_input = listen_and_execute()
    intent, slots = predict_intent_slots(user_input)
    print("Intent:", intent)
    print("Slots:", slots)
    if intent == "Readmail":
        return redirect('read_mail', slots=slots)
    elif intent == "Sendmail":
        return send_mail(slots)
    else:
        return "Unknown intent"


@app.route('/send')
# def send_mail():
#     """
#     Send email using Gmail API
#     """
#     slots = request.args.get('slots') 
#     print(slots)
#     service = Create_Service(CLIENT_SECRET_FILE, API_NAME, API_VERSION, SCOPES)
   
#     # emailMsg = dictate_email_body()
#     # mimeMessage = MIMEMultipart()
#     # mimeMessage['to'] = 'vmail456345@gmail.com'
#     # mimeMessage['subject'] = 'You weree'
#     # mimeMessage.attach(MIMEText(emailMsg, 'plain'))
#     # raw_string = base64.urlsafe_b64encode(mimeMessage.as_bytes()).decode()

#     # message = service.users().messages().send(userId='me', body={'raw': raw_string}).execute()
#     return 'Email sent successfully!'

@app.route('/read')

def read():
    read_emails()


@app.route('/att')
def att():
    result =send_email_att()
    return result
  

@app.route('/delete')      
def delete():
    sender_name = "vmail456345@gmail.com"
    result = delete_last_message_from_sender(sender_name)
    return result

@app.route('/search')      
def search():
    sender_name = "noreply@jobalertshub.com"
    search_email(sender_name)


@app.route('/draft')
def draftemail():
    result = create_draft_email()
    return result
    

if __name__ == '__main__':
    app.run()
