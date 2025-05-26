CREATE TABLE public.queimadas (
    id UUID PRIMARY KEY,
    lat DOUBLE PRECISION NULL,
    lon DOUBLE PRECISION NULL,
    data_hora_gmt TIMESTAMP WITHOUT TIME ZONE NULL,
    satelite TEXT NULL,
    municipio TEXT NULL,
    estado TEXT NULL,
    pais TEXT NULL,
    municipio_id INTEGER,
    estado_id INTEGER,
    pais_id INTEGER,
    numero_dias_sem_chuva INTEGER,
    precipitacao DOUBLE PRECISION,
    risco_fogo DOUBLE PRECISION,
    bioma TEXT,
    frp DOUBLE PRECISION
);