CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE,
    password VARCHAR(100) NOT NULL,
    role ENUM('admin', 'doctor') NOT NULL
);
INSERT INTO users (username, password, role) VALUES
('admin1', 'admin1', 'admin'),
('dr.shaker', 'dr.shaker', 'doctor');
