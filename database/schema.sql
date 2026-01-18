-- Appointments Table Schema for PostgreSQL

CREATE TABLE IF NOT EXISTS appointments (
    id SERIAL PRIMARY KEY,
    customer_name VARCHAR(255) NOT NULL,
    address TEXT,
    appointment_title VARCHAR(255) NOT NULL,
    notes TEXT,
    start_time TIMESTAMP NOT NULL,
    estimated_end_time TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_end_time CHECK (estimated_end_time > start_time)
);

-- Create index for faster queries on start_time
CREATE INDEX idx_appointments_start_time ON appointments(start_time);

-- Create index for customer name lookups
CREATE INDEX idx_appointments_customer_name ON appointments(customer_name);