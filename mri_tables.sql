-- MRI Scans Database Tables
-- This file provides scripts for both MySQL and SQL Server (T-SQL)
-- Run the appropriate script based on your database system

-- =========================================================================
-- MYSQL VERSION - Use this if your database is MySQL
-- =========================================================================

-- Create table for storing MRI scan metadata
CREATE TABLE mri_scans (
    scan_id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT NOT NULL,
    scan_date DATETIME NOT NULL,
    scan_type ENUM('T1', 'T2', 'FLAIR', 'DWI', 'OTHER') NOT NULL,
    scan_description TEXT,
    file_path VARCHAR(255) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size INT,
    uploaded_by INT,
    uploaded_at DATETIME NOT NULL,
    is_processed TINYINT(1) DEFAULT 0,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE,
    FOREIGN KEY (uploaded_by) REFERENCES users(id) ON DELETE SET NULL
);

-- Create table for storing MRI scan processing results
CREATE TABLE mri_processing_results (
    result_id INT AUTO_INCREMENT PRIMARY KEY,
    scan_id INT NOT NULL,
    processing_date DATETIME NOT NULL,
    processor_type ENUM('CNN', 'SWIN', 'OTHER') NOT NULL,
    prediction_result VARCHAR(50),
    confidence_score DECIMAL(5,4),
    heatmap_path VARCHAR(255),
    features_extracted TEXT,
    processing_notes TEXT,
    FOREIGN KEY (scan_id) REFERENCES mri_scans(scan_id) ON DELETE CASCADE
);

-- Create table for linking MRI scans to Alzheimer's analyses
CREATE TABLE mri_alzheimer_analysis (
    id INT AUTO_INCREMENT PRIMARY KEY,
    scan_id INT NOT NULL,
    analysis_id INT NOT NULL,
    used_in_prediction TINYINT(1) DEFAULT 0,
    notes TEXT,
    FOREIGN KEY (scan_id) REFERENCES mri_scans(scan_id) ON DELETE CASCADE,
    FOREIGN KEY (analysis_id) REFERENCES alzheimers_analysis(analysis_id) ON DELETE CASCADE
);

-- Create table for storing MRI regions of interest measurements
CREATE TABLE mri_roi_measurements (
    measurement_id INT AUTO_INCREMENT PRIMARY KEY,
    scan_id INT NOT NULL,
    region_name VARCHAR(100) NOT NULL,
    volume_mm3 DECIMAL(10,2),
    thickness_mm DECIMAL(7,4),
    intensity_mean DECIMAL(8,4),
    notes TEXT,
    FOREIGN KEY (scan_id) REFERENCES mri_scans(scan_id) ON DELETE CASCADE
);

-- Add indexes for performance
CREATE INDEX idx_mri_scans_patient ON mri_scans(patient_id);
CREATE INDEX idx_mri_results_scan ON mri_processing_results(scan_id);
CREATE INDEX idx_mri_alzheimer_scan ON mri_alzheimer_analysis(scan_id);
CREATE INDEX idx_mri_alzheimer_analysis ON mri_alzheimer_analysis(analysis_id);


-- =========================================================================
-- SQL SERVER VERSION - Use this if your database is SQL Server
-- =========================================================================
/*
-- Create table for storing MRI scan metadata
CREATE TABLE mri_scans (
    scan_id INT IDENTITY(1,1) PRIMARY KEY,
    patient_id INT NOT NULL,
    scan_date DATETIME NOT NULL,
    scan_type VARCHAR(10) NOT NULL,
    scan_description TEXT,
    file_path VARCHAR(255) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size INT,
    uploaded_by INT,
    uploaded_at DATETIME NOT NULL,
    is_processed BIT DEFAULT 0,
    CONSTRAINT FK_mri_scans_patient FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    CONSTRAINT FK_mri_scans_user FOREIGN KEY (uploaded_by) REFERENCES users(id)
);

-- Create table for storing MRI scan processing results
CREATE TABLE mri_processing_results (
    result_id INT IDENTITY(1,1) PRIMARY KEY,
    scan_id INT NOT NULL,
    processing_date DATETIME NOT NULL,
    processor_type VARCHAR(10) NOT NULL,
    prediction_result VARCHAR(50),
    confidence_score DECIMAL(5,4),
    heatmap_path VARCHAR(255),
    features_extracted TEXT,
    processing_notes TEXT,
    CONSTRAINT FK_mri_results_scan FOREIGN KEY (scan_id) REFERENCES mri_scans(scan_id)
);

-- Create table for linking MRI scans to Alzheimer's analyses
CREATE TABLE mri_alzheimer_analysis (
    id INT IDENTITY(1,1) PRIMARY KEY,
    scan_id INT NOT NULL,
    analysis_id INT NOT NULL,
    used_in_prediction BIT DEFAULT 0,
    notes TEXT,
    CONSTRAINT FK_mri_alzheimer_scan FOREIGN KEY (scan_id) REFERENCES mri_scans(scan_id),
    CONSTRAINT FK_mri_alzheimer_analysis FOREIGN KEY (analysis_id) REFERENCES alzheimers_analysis(analysis_id)
);

-- Create table for storing MRI regions of interest measurements
CREATE TABLE mri_roi_measurements (
    measurement_id INT IDENTITY(1,1) PRIMARY KEY,
    scan_id INT NOT NULL,
    region_name VARCHAR(100) NOT NULL,
    volume_mm3 DECIMAL(10,2),
    thickness_mm DECIMAL(7,4),
    intensity_mean DECIMAL(8,4),
    notes TEXT,
    CONSTRAINT FK_mri_roi_scan FOREIGN KEY (scan_id) REFERENCES mri_scans(scan_id)
);

-- Add indexes for performance
CREATE INDEX idx_mri_scans_patient ON mri_scans(patient_id);
CREATE INDEX idx_mri_results_scan ON mri_processing_results(scan_id);
CREATE INDEX idx_mri_alzheimer_scan ON mri_alzheimer_analysis(scan_id);
CREATE INDEX idx_mri_alzheimer_analysis ON mri_alzheimer_analysis(analysis_id);
*/ 