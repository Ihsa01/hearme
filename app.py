from flask import Flask,redirect, url_for, request,render_template
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
from routes.read_mail import read_mail
from routes.delete_mail import delete_mail
from routes.draft_mail import draft_mail
from routes.search_mail import search_mail
import cv2
from flask import Flask,jsonify,request,render_template
import numpy as np
import face_recognition
from flask import Flask, jsonify, request




#speech to text
    
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()




#texttospeech
    
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
        model = JointIntentAndSlotFillingModel(total_intent_no=5, total_slot_no=9, dropout_prob=0.1)
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



from routes.send_mail import send_mail_bp
from routes.read_mail import read_mail_bp
from routes.delete_mail import delete_mail_bp
from routes.draft_mail import draft_mail_bp
from routes.search_mail import search_mail_bp
app.register_blueprint(search_mail_bp)
app.register_blueprint(delete_mail_bp)
app.register_blueprint(send_mail_bp)
app.register_blueprint(read_mail_bp)
app.register_blueprint(draft_mail_bp)


@app.route('/add')
def add():
   
    flag=1
    while(flag):
        # Get user input through speech recognition
        speak("speak")
        user_input = listen_and_execute()
        
        print(user_input)
        speak("if correct say yes")
        con=listen_and_execute()
        # Check if user confirms the input
        if 'yes' in con:
            if "add contact" in user_input:
                    speak("Speak username")
                    command=listen_and_execute()
                    username = command
                    speak("Speak email")
                    email = ""
                    recognizer=sr.Recognizer()
                    with sr.Microphone() as source:
                        print("listening")
                        email_audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
                        letter = recognizer.recognize_google(email_audio).lower()
                        print("Letter:", letter)
                        # elif letter in number:
                        #     letter = str(number[letter])
                        email += letter
                        email = email.replace(" ", "")
                        print(email)
                
                    
                    if not username or not email:
                        return 'Both username and email are required'


                    user = Users(username=username, email=email)
                    db.session.add(user)
                    db.session.commit()
                    speak("'User added successfully'")
                    return 'User added successfully'
                
            else:
                intent, slots = predict_intent_slots(user_input)
                print("Intent:", intent)
                print("Slots:", slots)
                flag=0
                if intent == "Draftmail":
                    
                    reciever_name = slots.get('B-reciever')
                    b_subject = slots.get('B-subject')
                    i_subject = slots.get('I-subject')
                    print(b_subject, i_subject)
                    if b_subject is not None and i_subject is not None:
                        # Concatenate the values of 'B-subject' and 'I-subject' with a space between them
                        subject = b_subject + ' ' + i_subject
                    elif b_subject is not None:
                        # Use the value of 'B-subject' if only it exists
                        subject = b_subject
                    elif i_subject is not None:
                        # Use the value of 'I-subject' if only it exists
                        subject = i_subject
                    else:
                        subject=None
                    return draft_mail(subject)
                # Check the intent and redirect accordingly
                elif intent == "Readmail":
                    sender_name = slots.get('B-sender')
                    if sender_name:
                        with app.app_context():
                    # Access the db to reflect the tables
                            db.reflect()
                            sender = Users.query.filter_by(username=sender_name).first()
                            if sender:
                                sender_email = sender.email
                            else:
                                return speak("sender not found in database")

                            print(sender_email)
                            return read_mail(sender_email)
                    else:
                        return read_mail()
                elif intent == "Searchmail":
                    sender_name = slots.get('B-sender')
                    
                    b_subject = slots.get('B-subject')
                    i_subject = slots.get('I-subject')
                    if b_subject is not None and i_subject is not None:
                        # Concatenate the values of 'B-subject' and 'I-subject' with a space between them
                        subject = b_subject + ' ' + i_subject
                    elif b_subject is not None:
                        # Use the value of 'B-subject' if only it exists
                        subject = b_subject
                    elif i_subject is not None:
                        # Use the value of 'I-subject' if only it exists
                        subject = i_subject
                    if sender_name:
                        with app.app_context():
                    # Access the db to reflect the tables
                            db.reflect()
                            sender = Users.query.filter_by(username=sender_name).first()
                            if sender:
                                sender_email = sender.email
                            else:
                                return speak("sender not found in database")

                            print(sender_email,subject)
                            if subject is None:
                                return read_mail(sender_email)
                            else:
                                return search_mail(sender_email,subject)
                    else:
                        sender_email=None
                        return search_mail(sender_email,subject)    


                elif intent == "Sendmail":
                    reciever_name = slots.get('B-reciever')
                    subject=slots.get('B-subject')
                    with app.app_context():
                    # Access the db to reflect the tables
                            db.reflect()
                            reciever = Users.query.filter_by(username=reciever_name).first()
                            if reciever:
                                reciever_email = reciever.email
                            else:
                                return speak("reciever not found in database")

                            print(reciever_email)
                    return send_mail(reciever_email, subject)
                elif intent == "deletemail":
                        sender_name = slots.get('B-sender')
                        with app.app_context():
                        # Access the db to reflect the tables
                            db.reflect()
                            sender = Users.query.filter_by(username=sender_name).first()
                            if sender:
                                sender_email = sender.email
                            else:
                                return speak("sender not found in database")

                            print(sender_email)
                        return delete_mail(sender_email)
                else:
                    return "Unknown intent"
        



registered_data = {}

registration_status_file = "registration_status.txt"


# Function to read registration status from file
def read_registration_status():
    if os.path.exists(registration_status_file):
        with open(registration_status_file, "r") as file:
            return file.read().strip() == "done"
    return False

# Function to write registration status to file
def write_registration_status(status):
    with open(registration_status_file, "w") as file:
        file.write("done" if status else "not_done")

# Check if registration has been done
registration_done = read_registration_status()

@app.route("/") 
def index():
    if registration_done:
        return redirect(url_for('login'))
    else:
        return render_template("index.html")

@app.route("/register",methods=["POST"])
def register():
    name=listen_and_execute()
    #name = request.form.get("name")
    photo = request.files['photo']

    uploads_folder = os.path.join(os.getcwd(),"static","uploads") 

    if not os.path.exists(uploads_folder):
        os.makedirs(uploads_folder)

    global registration_done
    if registration_done:
        return redirect(url_for('login'))

    photo.save(os.path.join(uploads_folder,f'{name}.jpg'))
    registered_data[name] = f"{name}.jpg"

    registration_done = True
    write_registration_status(True)

    response = {"success":True,'name':name}
    return jsonify(response)

@app.route("/login")

def login():

    video_capture = cv2.VideoCapture(0)
    known_face_encodings = []
    known_face_names = []

# Path to the folder containing the images of known faces
    uploads_folder = os.path.join(os.getcwd(),"static","uploads")

# Iterate over files in the uploads folder
    for filename in os.listdir(uploads_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load the image file
            image_path = os.path.join(uploads_folder, filename)
            image = face_recognition.load_image_file(image_path)

            # Find face encodings for all faces in the image
            face_encodings = face_recognition.face_encodings(image)

            # Add each face encoding and its corresponding filename to the lists
            for face_encoding in face_encodings:
                known_face_encodings.append(face_encoding)
                known_face_names.append(os.path.splitext(filename)[0])

    print("Known faces loaded successfully.")

    # Now you have the known_face_encodings and known_face_names lists populated with face encodings and their corresponding filenames.

    
    
    face_locations = []
    face_encodings = []
    face_names = []
    s=True
 
 
 
 
    import time  # Import the time module

# Initialize variables for face tracking
    current_user_name = filename
    basename = os.path.basename(current_user_name)
    current_user = os.path.splitext(basename)[0]
    print(current_user)
    user_present_start_time = None
    min_detection_time = 5  # Minimum time (in seconds) for face to be continuously detected

    while True:
        ret, frame = video_capture.read()  # Capture a frame and check if it's successful
        if not ret:
             print("Error: Failed to capture frame from webcam.")
             break
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])
        if s:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = ""
                face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distance)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    
                if(face_distance[best_match_index]>0.50):
                    return "unsuccesfull"
                # If the detected face is the same as the current user
                if name == current_user:
                    print(current_user)
                    # If user_present_start_time is not set, set it
                    if user_present_start_time is None:
                        user_present_start_time = time.time()
                    # If the face has been continuously detected for more than min_detection_time seconds
                    elif time.time() - user_present_start_time >= min_detection_time:
                        
                        video_capture.release()
                        cv2.destroyAllWindows()
                        user_present_start_time = time.time()
                        print("login succesfully")
                        return render_template('indexx.html')   
                        
                        # Update the user_present_start_time to track continuous detection
                        
                else:
                    # If a different face is detected, reset the current_user and user_present_start_time
                    return "unsuccesfull"
                   
               
                    
        cv2.imshow("authenticate", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return "Error: Login failed"

        



@app.route("/speak", methods=["POST"])
def speak_endpoint():
    data = request.json
    text = data.get("text")
    speak(text)
    return jsonify({"success": True})
   

if __name__ == '__main__':
    app.run()
