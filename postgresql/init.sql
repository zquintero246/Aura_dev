CREATE DATABASE aura_main;

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(150) UNIQUE,
    google_id VARCHAR(150),
    github_id VARCHAR(150),
    avatar VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE personal_access_tokens (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(64) UNIQUE,
    abilities TEXT,
    last_used_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabla principal para Home Assistant (si no existe)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'homes'
    ) THEN
        CREATE TABLE public.homes (
            user_id VARCHAR(255) PRIMARY KEY,
            city VARCHAR(255),
            country VARCHAR(255),
            lat NUMERIC(10,6),
            lon NUMERIC(10,6),
            created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_homes_country ON public.homes(country);
        CREATE INDEX IF NOT EXISTS idx_homes_city ON public.homes(city);
    END IF;
END$$;
