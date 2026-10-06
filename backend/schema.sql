-- Retina Guard AI - MySQL schema
-- Run with:  mysql -u root -p < schema.sql

CREATE DATABASE IF NOT EXISTS retina_guard;
USE retina_guard;

CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    filename VARCHAR(255),
    grade VARCHAR(50),
    confidence FLOAT,
    lesion_count INT NULL,
    mode VARCHAR(30),
    created_at DATETIME
);
