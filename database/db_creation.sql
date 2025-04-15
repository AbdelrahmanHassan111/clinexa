CREATE DATABASE smart_clinic;
USE smart_clinic;
CREATE TABLE patients (
    patient_id INT PRIMARY KEY AUTO_INCREMENT,
    full_name VARCHAR(100),
    gender ENUM('Male', 'Female', 'Other'),
    birth_date DATE,
    contact_info VARCHAR(100),
    address TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE doctors (
    doctor_id INT PRIMARY KEY AUTO_INCREMENT,
    full_name VARCHAR(100),
    specialization VARCHAR(100),
    email VARCHAR(100),
    phone_number VARCHAR(20)
);
CREATE TABLE medical_records (
    record_id INT PRIMARY KEY AUTO_INCREMENT,
    patient_id INT,
    diagnosis TEXT,
    visit_date DATE,
    notes TEXT,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);
CREATE TABLE alzheimers_analysis (
    analysis_id INT PRIMARY KEY AUTO_INCREMENT,
    patient_id INT,
    input_features JSON,  -- Store all numerical input values in JSON or structured format
    prediction VARCHAR(50),
    confidence_score FLOAT,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);
CREATE TABLE appointments (
    appointment_id INT PRIMARY KEY AUTO_INCREMENT,
    patient_id INT,
    doctor_id INT,
    appointment_date DATETIME,
    reason TEXT,
    status ENUM('Scheduled', 'Completed', 'Cancelled'),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id)
);
CREATE TABLE chat_logs (
    chat_id INT PRIMARY KEY AUTO_INCREMENT,
    patient_id INT,
    doctor_id INT,
    message TEXT,
    sender ENUM('doctor', 'model'),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id)
);
INSERT INTO patients (full_name, gender, birth_date, contact_info)
VALUES ('Ahmed Youssef', 'Male', '1950-03-12', 'ahmed@example.com');

INSERT INTO doctors (full_name, specialization, email)
VALUES ('Dr. Mariam Hossam', 'Neurologist', 'mariam@clinic.com');
