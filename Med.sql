CREATE DATABASE IF NOT EXISTS medical_db;

USE medical_db;


CREATE TABLE IF NOT EXISTS patients (
    patient_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    age INT,
    gender VARCHAR(10),
    diagnosis TEXT,
    address VARCHAR(255),
    medicines TEXT,
    medical_history TEXT
);

-- New Appointments table
CREATE TABLE IF NOT EXISTS appointments (
    appointment_id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    appointment_date DATE NOT NULL,
    appointment_time TIME NOT NULL,
    reason TEXT,
    status ENUM('Scheduled', 'Completed', 'Cancelled') DEFAULT 'Scheduled',
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- Create a view for easier appointment management
CREATE VIEW appointment_details AS
SELECT 
    a.appointment_id,
    a.patient_id,
    p.name AS patient_name,
    a.appointment_date,
    a.appointment_time,
    a.reason,
    a.status
FROM appointments a
LEFT JOIN patients p ON a.patient_id = p.patient_id;
SELECT * FROM patients;
SELECT * FROM appointments;