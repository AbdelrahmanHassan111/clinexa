CREATE TABLE IF NOT EXISTS patient_diagnosis_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    CDRSB FLOAT,
    mPACCdigit FLOAT,
    MMSE FLOAT,
    LDELTOTAL FLOAT,
    -- ... all other features here ...
    TAU FLOAT,
    prediction_result VARCHAR(50),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);
