from flask import Flask, request, render_template, redirect, url_for, jsonify
import mysql.connector
from datetime import datetime, time, timedelta
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import joblib
import numpy as np
import os
from threading import Thread



openai.api_key = os.getenv('OPENAI_API_KEY', '')

app = Flask(__name__)

# Configure MySQL
db = mysql.connector.connect(
    host="127.0.0.1",
    user="root",
    password="Ayush#3078",
    database="medical_db"
)
cursor = db.cursor(dictionary=True)

# Configure OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize Medicine Recommendation Model
def init_medicine_model():
    data = {
        'diagnosis': ['flu', 'hypertension', 'diabetes', 'migraine', 'asthma'],
        'medicines': [
            'Paracetamol 500mg, Ibuprofen 200mg',
            'Lisinopril 10mg, Amlodipine 5mg',
            'Metformin 500mg, Insulin',
            'Sumatriptan 50mg, Ibuprofen 400mg',
            'Albuterol inhaler, Prednisone'
        ]
    }
    df = pd.DataFrame(data)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['diagnosis'])
    model = NearestNeighbors(n_neighbors=1)
    model.fit(X)
    joblib.dump((model, vectorizer, df), 'medicine_model.pkl')

try:
    medicine_model, medicine_vectorizer, medicine_df = joblib.load('medicine_model.pkl')
except:
    init_medicine_model()
    medicine_model, medicine_vectorizer, medicine_df = joblib.load('medicine_model.pkl')

# Existing Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = {field: request.form[field] for field in ['name', 'age', 'gender', 'diagnosis', 'address', 'medicines', 'medical_history']}
        cursor.execute("""
            INSERT INTO patients (name, age, gender, diagnosis, address, medicines, medical_history) 
            VALUES (%(name)s, %(age)s, %(gender)s, %(diagnosis)s, %(address)s, %(medicines)s, %(medical_history)s)
        """, data)
        db.commit()
        return "Patient added successfully!"
    return render_template('form.html')

@app.route('/view', methods=['GET', 'POST'])
def view():
    patient = None
    if request.method == 'POST':
        cursor.execute("SELECT * FROM patients WHERE patient_id = %s", (request.form['patient_id'],))
        patient = cursor.fetchone()
    return render_template('view.html', patient=patient)

@app.route('/appointments', methods=['GET', 'POST'])
def appointments():
    if request.method == 'POST':
        data = {
            'patient_id': request.form['patient_id'],
            'appointment_date': datetime.strptime(request.form['appointment_date'], '%Y-%m-%d').date(),
            'appointment_time': datetime.strptime(request.form['appointment_time'], '%H:%M').time(),
            'reason': request.form['reason']
        }
        cursor.execute("""
            INSERT INTO appointments (patient_id, appointment_date, appointment_time, reason)
            VALUES (%(patient_id)s, %(appointment_date)s, %(appointment_time)s, %(reason)s)
        """, data)
        db.commit()
        return redirect(url_for('view_appointments'))
    
    cursor.execute("SELECT patient_id, name FROM patients")
    return render_template('appointments.html', patients=cursor.fetchall())

@app.route('/view-appointments')
def view_appointments():
    cursor.execute("""
        SELECT a.*, p.name AS patient_name 
        FROM appointments a
        JOIN patients p ON a.patient_id = p.patient_id
        ORDER BY a.appointment_date, a.appointment_time
    """)
    appointments = cursor.fetchall()
    
    for appt in appointments:
        if isinstance(appt['appointment_time'], timedelta):
            seconds = appt['appointment_time'].total_seconds()
            appt['appointment_time'] = time(int(seconds // 3600), int((seconds % 3600) // 60))
    
    return render_template('view_appointments.html', appointments=appointments)

# AI Routes
# Updated AI Assistant Route
@app.route('/ai-assistant', methods=['GET', 'POST'])
def ai_assistant():
    cursor.execute("SELECT patient_id, name FROM patients")
    patients = cursor.fetchall()
    
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        patient_id = request.form.get('patient_id')
        
        if not question:
            return render_template('ai_assistant.html', 
                               patients=patients,
                               error="Please enter a question")
        
        try:
            # Build context-aware prompt
            prompt = f"You are a medical assistant. Answer this professionally:\n\nQuestion: {question}"
            
            if patient_id:
                cursor.execute("SELECT * FROM patients WHERE patient_id = %s", (patient_id,))
                patient = cursor.fetchone()
                if patient:
                    prompt = f"""Patient Context:
- Name: {patient['name']}
- Age: {patient['age']}
- Diagnosis: {patient['diagnosis']}
- Medications: {patient['medicines']}
- History: {patient['medical_history']}

Question: {question}"""
            
            # Get AI response
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "system",
                    "content": "You are a helpful medical assistant. Provide accurate, concise answers."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                temperature=0.5,
                max_tokens=500
            )
            
            ai_response = response.choices[0].message.content
            
            return render_template('ai_assistant.html',
                               patients=patients,
                               response=ai_response,
                               question=question,
                               selected_patient=patient_id)
            
        except Exception as e:
            return render_template('ai_assistant.html',
                               patients=patients,
                               error=f"AI Service Error: {str(e)}")
    
    return render_template('ai_assistant.html', patients=patients)

@app.route('/api/predict-medicine', methods=['POST'])
def predict_medicine():
    diagnosis = request.json.get('diagnosis', '')
    X = medicine_vectorizer.transform([diagnosis])
    distances, indices = medicine_model.kneighbors(X)
    return jsonify({
        'medicines': medicine_df.iloc[indices[0][0]]['medicines'],
        'confidence': float(1 - distances[0][0])
    })

@app.route('/api/predict-recovery', methods=['POST'])
def predict_recovery():
    diagnosis = request.json.get('diagnosis', '').lower()
    age = int(request.json.get('age', 30))
    
    recovery_data = {
        'flu': 7, 'hypertension': 30, 'diabetes': 90,
        'migraine': 1, 'asthma': 2
    }
    days = recovery_data.get(diagnosis, 14) * (1.1 if age > 50 else 0.9)
    return jsonify({
        'days': int(days),
        'date': (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
    })

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error=str(error)), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)