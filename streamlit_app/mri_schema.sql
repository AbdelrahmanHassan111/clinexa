-- MRI Scans Table
CREATE TABLE IF NOT EXISTS mri_scans (
    scan_id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT NOT NULL,
    scan_date DATETIME NOT NULL,
    scan_type VARCHAR(50) NOT NULL,
    file_path VARCHAR(255) NOT NULL,
    is_processed BOOLEAN DEFAULT FALSE,
    prediction VARCHAR(100) NULL,
    confidence FLOAT NULL,
    scan_notes TEXT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

-- MRI ROI Measurements Table
CREATE TABLE IF NOT EXISTS mri_roi_measurements (
    measurement_id INT AUTO_INCREMENT PRIMARY KEY,
    scan_id INT NOT NULL,
    measurement_date DATETIME NOT NULL,
    hippocampus_left FLOAT NULL,
    hippocampus_right FLOAT NULL,
    hippocampus_total FLOAT NULL,
    entorhinal_left FLOAT NULL,
    entorhinal_right FLOAT NULL,
    entorhinal_total FLOAT NULL,
    lateral_ventricles FLOAT NULL,
    whole_brain FLOAT NULL,
    temporal_lobe_left FLOAT NULL,
    temporal_lobe_right FLOAT NULL,
    temporal_lobe_total FLOAT NULL,
    fusiform_left FLOAT NULL,
    fusiform_right FLOAT NULL,
    fusiform_total FLOAT NULL,
    amygdala_left FLOAT NULL,
    amygdala_right FLOAT NULL,
    amygdala_total FLOAT NULL,
    total_intracranial_volume FLOAT NULL,
    normalized_values JSON NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (scan_id) REFERENCES mri_scans(scan_id) ON DELETE CASCADE
); 