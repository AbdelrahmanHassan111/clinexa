-- Update the users table to include patient_id field
ALTER TABLE users ADD COLUMN IF NOT EXISTS patient_id INT NULL;
ALTER TABLE users ADD CONSTRAINT fk_user_patient FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE ON UPDATE CASCADE;

-- Update the patients table with new fields
ALTER TABLE patients 
ADD COLUMN IF NOT EXISTS email VARCHAR(100) NULL,
ADD COLUMN IF NOT EXISTS emergency_contact VARCHAR(100) NULL,
ADD COLUMN IF NOT EXISTS emergency_phone VARCHAR(20) NULL,
ADD COLUMN IF NOT EXISTS allergies TEXT NULL,
ADD COLUMN IF NOT EXISTS medical_conditions TEXT NULL;

-- Add unique constraint on email
ALTER TABLE patients ADD CONSTRAINT unique_patient_email UNIQUE (email);

-- Update the appointments table
ALTER TABLE appointments 
ADD COLUMN IF NOT EXISTS created_at DATETIME NULL,
ADD COLUMN IF NOT EXISTS cancellation_date DATETIME NULL; 