import sklearn
import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template, jsonify
from collections import defaultdict
from textblob import TextBlob

# flask app
app = Flask(__name__)

# load databasedataset===================================
sym_des = pd.read_csv("dataset/symtoms_df.csv")
precautions = pd.read_csv("dataset/precautions_df.csv")
workout = pd.read_csv("dataset/workout_df.csv")
description = pd.read_csv("dataset/description.csv")
medications = pd.read_csv('dataset/medications.csv')
diets = pd.read_csv("dataset/diets.csv")

# load model===========================================
svc = pickle.load(open('model/svc.pkl','rb'))

#============================================================
# custome and helping functions
#==========================helper funtions================

def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']

    return desc,pre,med,die,wrkout

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# # Initialize the TextBlob object for spelling correction
def correct_spelling(symptom):
    # Correct the spelling of a single symptom
    blob = TextBlob(symptom)
    return str(blob.correct())

symptom_mapping = defaultdict(lambda: "unknown", {
    "itching": ["itching"],
    "skin_rash": ["skin rash", "rash", "dermatitis"],
    "nodal_skin_eruptions": ["nodal skin eruptions", "skin eruptions", "bumps"],
    "continuous_sneezing": ["continuous sneezing", "sneezing"],
    "shivering": ["shivering", "trembling"],
    "chills": ["chills", "cold sensation", "cold"],
    "joint_pain": ["joint pain", "arthralgia", "aching joints"],
    "stomach_pain": ["stomach pain", "abdominal pain", "belly ache"],
    "acidity": ["acidity", "heartburn", "acid reflux"],
    "ulcers_on_tongue": ["ulcers on tongue", "tongue ulcers", "mouth sores"],
    "muscle_wasting": ["muscle wasting", "muscle loss"],
    "vomiting": ["vomiting", "emesis", "throwing up"],
    "burning_micturition": ["burning micturition", "burning urination", "painful urination"],
    "spotting_urination": ["spotting urination", "blood in urine", "hematuria"],
    "fatigue": ["fatigue", "tiredness", "exhaustion"],
    "weight_gain": ["weight gain", "increased weight"],
    "anxiety": ["anxiety", "nervousness", "worry"],
    "cold_hands_and_feets": ["cold hands and feet", "cold extremities"],
    "mood_swings": ["mood swings", "emotional changes"],
    "weight_loss": ["weight loss", "decreased weight"],
    "restlessness": ["restlessness", "agitation"],
    "lethargy": ["lethargy", "sluggishness"],
    "patches_in_throat": ["patches in throat", "throat patches", "throat lesions"],
    "irregular_sugar_level": ["irregular sugar level", "unstable glucose", "blood sugar fluctuations"],
    "cough": ["cough", "coughing"],
    "high_fever": ["high fever", "elevated temperature"],
    "sunken_eyes": ["sunken eyes", "hollow eyes"],
    "breathlessness": ["breathlessness", "shortness of breath", "dyspnea"],
    "sweating": ["sweating", "perspiration"],
    "dehydration": ["dehydration", "fluid loss"],
    "indigestion": ["indigestion", "upset stomach"],
    "headache": ["headache", "head pain", "migraine"],
    "yellowish_skin": ["yellowish skin", "jaundice"],
    "dark_urine": ["dark urine"],
    "nausea": ["nausea", "queasiness"],
    "loss_of_appetite": ["loss of appetite", "no appetite", "anorexia"],
    "pain_behind_the_eyes": ["pain behind the eyes", "eye pain"],
    "back_pain": ["back pain", "lower back pain"],
    "constipation": ["constipation", "difficulty passing stool"],
    "abdominal_pain": ["abdominal pain", "belly pain", "stomach ache"],
    "diarrhoea": ["diarrhoea", "loose stools"],
    "mild_fever": ["mild fever", "fever" "low-grade fever"],
    "yellow_urine": ["yellow urine"],
    "yellowing_of_eyes": ["yellowing of eyes", "scleral icterus"],
    "acute_liver_failure": ["acute liver failure", "hepatic failure"],
    "fluid_overload": ["fluid overload", "edema"],
    "swelling_of_stomach": ["swelling of stomach", "abdominal bloating"],
    "swelled_lymph_nodes": ["swelled lymph nodes", "enlarged lymph nodes"],
    "malaise": ["malaise", "general discomfort"],
    "blurred_and_distorted_vision": ["blurred and distorted vision", "blurry vision"],
    "phlegm": ["phlegm", "mucus"],
    "throat_irritation": ["throat irritation", "sore throat"],
    "redness_of_eyes": ["redness of eyes", "bloodshot eyes"],
    "sinus_pressure": ["sinus pressure", "sinus congestion"],
    "runny_nose": ["runny nose", "rhinorrhea"],
    "congestion": ["congestion", "nasal blockage"],
    "chest_pain": ["chest pain", "angina"],
    "weakness_in_limbs": ["weakness in limbs", "limb weakness"],
    "fast_heart_rate": ["fast heart rate", "tachycardia"],
    "pain_during_bowel_movements": ["pain during bowel movements", "painful defecation"],
    "pain_in_anal_region": ["pain in anal region", "anal pain"],
    "bloody_stool": ["bloody stool", "rectal bleeding"],
    "irritation_in_anus": ["irritation in anus", "anal itching"],
    "neck_pain": ["neck pain", "cervical pain"],
    "dizziness": ["dizziness", "lightheadedness"],
    "cramps": ["cramps", "muscle cramps", "spasms"],
    "bruising": ["bruising", "hematoma"],
    "obesity": ["obesity", "overweight"],
    "swollen_legs": ["swollen legs", "leg edema"],
    "swollen_blood_vessels": ["swollen blood vessels", "varicose veins"],
    "puffy_face_and_eyes": ["puffy face and eyes", "facial swelling"],
    "enlarged_thyroid": ["enlarged thyroid", "goiter"],
    "brittle_nails": ["brittle nails", "weak nails"],
    "swollen_extremeties": ["swollen extremities", "swollen arms and legs"],
    "excessive_hunger": ["excessive hunger", "polyphagia"],
    "extra_marital_contacts": ["extra marital contacts", "multiple sexual partners"],
    "drying_and_tingling_lips": ["drying and tingling lips", "lip dryness"],
    "slurred_speech": ["slurred speech", "dysarthria"],
    "knee_pain": ["knee pain", "pain in the knees"],
    "hip_joint_pain": ["hip joint pain", "hip pain"],
    "muscle_weakness": ["muscle weakness", "muscle fatigue"],
    "stiff_neck": ["stiff neck", "neck stiffness"],
    "swelling_joints": ["swelling joints", "joint swelling"],
    "movement_stiffness": ["movement stiffness", "rigidity"],
    "spinning_movements": ["spinning movements", "vertigo"],
    "loss_of_balance": ["loss of balance", "balance problems"],
    "unsteadiness": ["unsteadiness", "lack of balance"],
    "weakness_of_one_body_side": ["weakness of one body side", "hemiparesis"],
    "loss_of_smell": ["loss of smell", "anosmia"],
    "bladder_discomfort": ["bladder discomfort", "bladder pain"],
    "foul_smell_of_urine": ["foul smell of urine", "smelly urine"],
    "continuous_feel_of_urine": ["continuous feel of urine", "urgency to urinate"],
    "passage_of_gases": ["passage of gases", "flatulence"],
    "internal_itching": ["internal itching"],
    "toxic_look_(typhos)": ["toxic look (typhos)", "septic appearance"],
    "depression": ["depression", "low mood"],
    "irritability": ["irritability", "easily annoyed"],
    "muscle_pain": ["muscle pain", "myalgia"],
    "altered_sensorium": ["altered sensorium", "confusion"],
    "red_spots_over_body": ["red spots over body", "rash with red spots"],
    "belly_pain": ["belly pain", "abdominal pain"],
    "abnormal_menstruation": ["abnormal menstruation", "irregular periods"],
    "dischromic_patches": ["dischromic patches", "skin discoloration"],
    "watering_from_eyes": ["watering from eyes", "teary eyes"],
    "increased_appetite": ["increased appetite", "hyperphagia"],
    "polyuria": ["polyuria", "excessive urination"],
    "family_history": ["family history", "genetic predisposition"],
    "mucoid_sputum": ["mucoid sputum", "mucus in sputum"],
    "rusty_sputum": ["rusty sputum", "blood-tinged sputum"],
    "lack_of_concentration": ["lack of concentration", "difficulty focusing"],
    "visual_disturbances": ["visual disturbances", "vision problems"],
    "receiving_blood_transfusion": ["receiving blood transfusion"],
    "receiving_unsterile_injections": ["receiving unster"]
    # Add more symptoms and their synonyms here
})

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    
    # Print the initial input vector
    print("Initial input vector:", input_vector)
    
    for item in patient_symptoms:
        # First, check if the symptom is in the symptoms_dict
        corrected_item = item  # Initial assumption, might be corrected later
        index = symptoms_dict.get(corrected_item, -1)
        
        if index != -1:
            # If found in symptoms_dict
            input_vector[index] = 1
            print(f"Symptom '{item}' is mapped to index {index} directly from symptoms_dict")
        else:
            # Check if the symptom is in the symptom_mapping
            found = False
            for key, synonyms in symptom_mapping.items():
                if item in synonyms:
                    index = symptoms_dict.get(key, -1)
                    if index != -1:
                        input_vector[index] = 1
                        print(f"Symptom '{item}' is mapped to synonym '{key}' with index {index}")
                        found = True
                        break
            
            if not found:
                # Correct spelling if not found in either dictionary
                corrected_item = correct_spelling(item)
                index = symptoms_dict.get(corrected_item, -1)
                
                if index != -1:
                    # If found in symptoms_dict after correction
                    input_vector[index] = 1
                    print(f"Symptom '{item}' corrected to '{corrected_item}' and mapped to index {index}")
                else:
                    # Check if the corrected symptom is in the symptom_mapping
                    found = False
                    for key, synonyms in symptom_mapping.items():
                        if corrected_item in synonyms:
                            index = symptoms_dict.get(key, -1)
                            if index != -1:
                                input_vector[index] = 1
                                print(f"Symptom '{item}' corrected to '{corrected_item}' and mapped to synonym '{key}' with index {index}")
                                found = True
                                break
                    if not found:
                        print(f"Warning: Symptom '{item}' with corrected spelling '{corrected_item}' is not in the symptoms dictionary or mapping")
    
    # Print the final input vector
    
    predicted_disease_index = svc.predict([input_vector])[0]
    predicted_disease = diseases_list.get(predicted_disease_index, "Unknown Disease")
    
    # return predicted_disease
    return diseases_list[svc.predict([input_vector])[0]]

# creating routes========================================


@app.route("/")
def index():
    return render_template("index.html")

# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        location = request.form.get('location')
        symptoms = request.form.get('symptoms')
        # mysysms = request.form.get('mysysms')
        # print(mysysms)
        print(symptoms)
        if symptoms =="Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('index.html', message=message)
        
        if not symptoms:
            message = "Please enter symptoms."
            return render_template('index.html', message=message) 
        
        else:

            # Split the user's input into a list of symptoms (assuming they are comma-separated)
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            # Remove any extra characters, if any
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            my_precautions = []
            for i in precautions:
                my_precautions.append(i)

            return render_template('index.html', name=name, age=age, location=location, symptoms=symptoms,  predicted_disease=predicted_disease, dis_des=dis_des,
                                   my_precautions=my_precautions[0], medications=medications, my_diet=rec_diet,
                                   workout=workout)

    return render_template('index.html')



# about view funtion and path
@app.route('/about')
def about():
    return render_template("about.html")
# contact view funtion and path
@app.route('/contact')
def contact():
    return render_template("contact.html")

# developer view funtion and path
@app.route('/developer')
def developer():
    return render_template("developer.html")

# about view funtion and path
@app.route('/blog')
def blog():
    return render_template("blog.html")

# about view funtion and path
@app.route('/upload',  methods=['GET', 'POST'])
def upload():
    return render_template("upload.html")

if __name__ == '__main__':

    app.run(debug=True)