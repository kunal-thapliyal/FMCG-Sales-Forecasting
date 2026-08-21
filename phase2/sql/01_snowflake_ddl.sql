-- =====================================================================
-- FMCG 2022-2024 Sales Forecasting — Snowflake Star Schema DDL
-- Grain: FACT_SALES = one row per (Date, SKU, Region, Channel)
-- pack_type varies day-to-day within a SKU-Region-Channel series, so it
-- lives on the FACT table (a fact attribute), not on DIM_PRODUCT.
-- =====================================================================

CREATE DATABASE IF NOT EXISTS FMCG_FORECASTING;
CREATE SCHEMA IF NOT EXISTS FMCG_FORECASTING.SALES;
USE DATABASE FMCG_FORECASTING;
USE SCHEMA SALES;

CREATE OR REPLACE TABLE DIM_DATE (
    DATE_KEY       DATE PRIMARY KEY,
    YEAR            SMALLINT,
    QUARTER         SMALLINT,
    MONTH           SMALLINT,
    DAY_OF_WEEK     SMALLINT,
    IS_WEEKEND      BOOLEAN,
    WEEK_OF_YEAR    SMALLINT
);

-- One row per SKU: static attributes (verified 1:1 with sku in source data)
CREATE OR REPLACE TABLE DIM_PRODUCT (
    SKU             VARCHAR(20) PRIMARY KEY,
    BRAND           VARCHAR(50) NOT NULL,
    CATEGORY        VARCHAR(50) NOT NULL,
    SEGMENT         VARCHAR(50) NOT NULL
);

CREATE OR REPLACE TABLE DIM_REGION (
    REGION_KEY      INTEGER AUTOINCREMENT PRIMARY KEY,
    REGION          VARCHAR(50) NOT NULL UNIQUE
);

CREATE OR REPLACE TABLE DIM_CHANNEL (
    CHANNEL_KEY     INTEGER AUTOINCREMENT PRIMARY KEY,
    CHANNEL         VARCHAR(50) NOT NULL UNIQUE
);

CREATE OR REPLACE TABLE DIM_PACK_TYPE (
    PACK_TYPE_KEY   INTEGER AUTOINCREMENT PRIMARY KEY,
    PACK_TYPE       VARCHAR(50) NOT NULL UNIQUE
);

-- ---------------------------------------------------------------------
-- Fact: one row per (Date, SKU, Region, Channel)
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE FACT_SALES (
    ROW_ID              BIGINT PRIMARY KEY,
    DATE_KEY             DATE NOT NULL REFERENCES DIM_DATE(DATE_KEY),
    SKU                    VARCHAR(20) NOT NULL REFERENCES DIM_PRODUCT(SKU),
    REGION_KEY              INTEGER NOT NULL REFERENCES DIM_REGION(REGION_KEY),
    CHANNEL_KEY               INTEGER NOT NULL REFERENCES DIM_CHANNEL(CHANNEL_KEY),
    PACK_TYPE_KEY               INTEGER NOT NULL REFERENCES DIM_PACK_TYPE(PACK_TYPE_KEY),
    PRICE_UNIT                    NUMBER(8,2),
    PROMOTION_FLAG                  BOOLEAN,
    DELIVERY_DAYS                     SMALLINT,
    STOCK_AVAILABLE                     INTEGER,
    DELIVERED_QTY                         INTEGER,
    UNITS_SOLD                              INTEGER,
    IS_ANOMALOUS_ROW                          BOOLEAN
);

ALTER TABLE FACT_SALES CLUSTER BY (DATE_KEY, SKU);

-- ---------------------------------------------------------------------
-- Staging table + load template
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE STG_FMCG_CLEANED (
    ROW_ID BIGINT, DATE_KEY DATE, SKU VARCHAR(20), BRAND VARCHAR(50),
    SEGMENT VARCHAR(50), CATEGORY VARCHAR(50), CHANNEL VARCHAR(50),
    REGION VARCHAR(50), PACK_TYPE VARCHAR(50), PRICE_UNIT NUMBER(8,2),
    PROMOTION_FLAG BOOLEAN, DELIVERY_DAYS SMALLINT, STOCK_AVAILABLE INTEGER,
    DELIVERED_QTY INTEGER, UNITS_SOLD INTEGER, IS_ANOMALOUS_ROW BOOLEAN,
    YEAR SMALLINT, MONTH SMALLINT, QUARTER SMALLINT, DAY_OF_WEEK SMALLINT,
    IS_WEEKEND BOOLEAN, DAY_OF_YEAR SMALLINT, WEEK_OF_YEAR SMALLINT
);

-- CREATE OR REPLACE FILE FORMAT CSV_FF TYPE='CSV' FIELD_DELIMITER=','
--     SKIP_HEADER=1 FIELD_OPTIONALLY_ENCLOSED_BY='"' NULL_IF=('','NaN');
-- COPY INTO STG_FMCG_CLEANED FROM @~/fmcg_stage/fmcg_cleaned.csv
--     FILE_FORMAT=(FORMAT_NAME=CSV_FF) ON_ERROR='ABORT_STATEMENT';
