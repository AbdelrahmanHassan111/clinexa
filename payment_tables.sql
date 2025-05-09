-- Payment and Invoice Tables for Smart Clinic

-- Create table for invoices
CREATE TABLE invoices (
    invoice_id INT IDENTITY(1,1) PRIMARY KEY,
    patient_id INT NOT NULL,
    doctor_id INT NOT NULL,
    created_at DATETIME NOT NULL,
    due_date DATE NOT NULL,
    total_amount DECIMAL(10, 2) NOT NULL,
    status NVARCHAR(20) NOT NULL DEFAULT 'Pending',
    notes TEXT,
    CONSTRAINT FK_invoice_patient FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
    CONSTRAINT FK_invoice_doctor FOREIGN KEY (doctor_id) REFERENCES doctors(doctor_id) ON DELETE CASCADE
);

-- Create table for invoice items
CREATE TABLE invoice_items (
    item_id INT IDENTITY(1,1) PRIMARY KEY,
    invoice_id INT NOT NULL,
    description NVARCHAR(255) NOT NULL,
    item_type NVARCHAR(20) NOT NULL,
    quantity INT NOT NULL DEFAULT 1,
    unit_price DECIMAL(10, 2) NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    CONSTRAINT FK_invoice_items_invoice FOREIGN KEY (invoice_id) REFERENCES invoices(invoice_id) ON DELETE CASCADE
);

-- Create table for payments
CREATE TABLE payments (
    payment_id INT IDENTITY(1,1) PRIMARY KEY,
    invoice_id INT NOT NULL,
    payment_date DATETIME NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    payment_method NVARCHAR(20) NOT NULL,
    reference_number NVARCHAR(100),
    notes TEXT,
    created_by INT,
    CONSTRAINT FK_payment_invoice FOREIGN KEY (invoice_id) REFERENCES invoices(invoice_id) ON DELETE CASCADE,
    CONSTRAINT FK_payment_user FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
);

-- Add indexes for performance
CREATE INDEX idx_invoice_patient ON invoices(patient_id);
CREATE INDEX idx_invoice_doctor ON invoices(doctor_id);
CREATE INDEX idx_invoice_status ON invoices(status);
CREATE INDEX idx_payment_invoice ON payments(invoice_id);
CREATE INDEX idx_invoice_items_invoice ON invoice_items(invoice_id);

-- MRI Scan Tables for Alzheimer's Disease Analysis

-- Create table for MRI scans
CREATE TABLE mri_scans (
    scan_id INT IDENTITY(1,1) PRIMARY KEY,
    patient_id INT NOT NULL,
    scan_date DATETIME NOT NULL,
    scan_type NVARCHAR(20) NOT NULL,
    scan_description TEXT,
    file_path NVARCHAR(255) NOT NULL,
    file_name NVARCHAR(255) NOT NULL,
    file_type NVARCHAR(50),
    file_size INT,
    uploaded_by INT,
    uploaded_at DATETIME NOT NULL,
    is_processed BIT DEFAULT 0,
    CONSTRAINT FK_mri_patient FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
    CONSTRAINT FK_mri_user FOREIGN KEY (uploaded_by) REFERENCES users(id) ON DELETE SET NULL
);

-- Create table for MRI processing results
CREATE TABLE mri_processing_results (
    result_id INT IDENTITY(1,1) PRIMARY KEY,
    scan_id INT NOT NULL,
    processing_date DATETIME NOT NULL,
    processor_type NVARCHAR(50) NOT NULL,
    prediction_result NVARCHAR(50),
    confidence_score DECIMAL(5, 4),
    heatmap_path NVARCHAR(255),
    features_extracted NVARCHAR(MAX),
    CONSTRAINT FK_mri_results_scan FOREIGN KEY (scan_id) REFERENCES mri_scans(scan_id) ON DELETE CASCADE
);

-- Create table for linking MRI scans to Alzheimer's analyses
CREATE TABLE mri_alzheimer_analysis (
    link_id INT IDENTITY(1,1) PRIMARY KEY,
    scan_id INT NOT NULL,
    analysis_id INT NOT NULL,
    used_in_prediction BIT DEFAULT 0,
    notes TEXT,
    CONSTRAINT FK_mri_link_scan FOREIGN KEY (scan_id) REFERENCES mri_scans(scan_id) ON DELETE CASCADE,
    CONSTRAINT FK_mri_link_analysis FOREIGN KEY (analysis_id) REFERENCES alzheimers_analysis(analysis_id) ON DELETE CASCADE
);

-- Create table for brain region measurements from MRI
CREATE TABLE mri_roi_measurements (
    measurement_id INT IDENTITY(1,1) PRIMARY KEY,
    scan_id INT NOT NULL,
    measurement_date DATETIME NOT NULL,
    
    -- Hippocampus volumes (mm³)
    hippocampus_left DECIMAL(10, 2),
    hippocampus_right DECIMAL(10, 2),
    hippocampus_total DECIMAL(10, 2),
    
    -- Entorhinal cortex volumes (mm³)
    entorhinal_left DECIMAL(10, 2),
    entorhinal_right DECIMAL(10, 2),
    entorhinal_total DECIMAL(10, 2),
    
    -- Ventricle volume (mm³)
    lateral_ventricles DECIMAL(10, 2),
    
    -- Whole brain volume (mm³)
    whole_brain DECIMAL(10, 2),
    
    -- Temporal lobe volumes (mm³)
    temporal_lobe_left DECIMAL(10, 2),
    temporal_lobe_right DECIMAL(10, 2),
    temporal_lobe_total DECIMAL(10, 2),
    
    -- Fusiform gyrus volumes (mm³)
    fusiform_left DECIMAL(10, 2),
    fusiform_right DECIMAL(10, 2),
    fusiform_total DECIMAL(10, 2),
    
    -- Amygdala volumes (mm³)
    amygdala_left DECIMAL(10, 2),
    amygdala_right DECIMAL(10, 2),
    amygdala_total DECIMAL(10, 2),
    
    -- Total intracranial volume (mm³)
    total_intracranial_volume DECIMAL(10, 2),
    
    -- Normalized values as JSON for flexibility
    normalized_values NVARCHAR(MAX),
    
    CONSTRAINT FK_roi_scan FOREIGN KEY (scan_id) REFERENCES mri_scans(scan_id) ON DELETE CASCADE
);

-- Add indexes for MRI tables
CREATE INDEX idx_mri_patient ON mri_scans(patient_id);
CREATE INDEX idx_mri_type ON mri_scans(scan_type);
CREATE INDEX idx_mri_processed ON mri_scans(is_processed);
CREATE INDEX idx_mri_result_scan ON mri_processing_results(scan_id);
CREATE INDEX idx_mri_alzheimer_analysis ON mri_alzheimer_analysis(analysis_id, scan_id);
CREATE INDEX idx_roi_scan ON mri_roi_measurements(scan_id); 