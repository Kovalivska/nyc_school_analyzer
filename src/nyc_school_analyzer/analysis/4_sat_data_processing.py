#!/usr/bin/env python3
"""
SAT Data Processing and Database Integration Script
Svitlana Kovalivska - NYC SAT Results Analysis
"""

import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
import warnings
from typing import Dict, List, Tuple, Any
import re

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def assess_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive data quality assessment function
    Returns detailed statistics about data quality issues
    """
    quality_report = {
        'shape': df.shape,
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicate_rows': df.duplicated().sum(),
        'column_stats': {}
    }
    
    # Analyze each column
    for col in df.columns:
        stats = {
            'dtype': str(df[col].dtype),
            'unique_values': df[col].nunique(),
            'sample_values': df[col].unique()[:10].tolist()
        }
        
        # Check for suspicious values
        if col in ['Num of SAT Test Takers', 'SAT Critical Reading Avg. Score', 
                  'SAT Math Avg. Score', 'SAT Writing Avg. Score', 'SAT Critical Readng Avg. Score']:
            # Look for non-numeric values in score columns
            non_numeric = df[col].astype(str).str.contains(r'[^0-9\.]', na=False).sum()
            stats['non_numeric_count'] = non_numeric
            stats['suspicious_values'] = df[col][df[col].astype(str).str.contains(r'[^0-9\.]', na=False)].unique().tolist()
            
        quality_report['column_stats'][col] = stats
    
    return quality_report

def comprehensive_data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive data cleaning function following data science best practices
    """
    print("=== STARTING DATA CLEANING PROCESS ===")
    df_clean = df.copy()
    
    # Step 1: Remove exact duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"Step 1: Removed {initial_rows - len(df_clean)} duplicate rows")
    
    # Step 2: Remove the duplicate column with typo
    if 'SAT Critical Readng Avg. Score' in df_clean.columns:
        df_clean = df_clean.drop('SAT Critical Readng Avg. Score', axis=1)
        print("Step 2: Removed duplicate column 'SAT Critical Readng Avg. Score'")
    
    # Step 3: Clean SAT score columns
    sat_score_columns = ['SAT Critical Reading Avg. Score', 'SAT Math Avg. Score', 'SAT Writing Avg. Score']
    
    for col in sat_score_columns:
        if col in df_clean.columns:
            # Replace 's' (suppressed) with NaN
            df_clean[col] = df_clean[col].replace('s', np.nan)
            
            # Convert to numeric
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Remove invalid scores (SAT scores should be between 200-800)
            valid_mask = (df_clean[col] >= 200) & (df_clean[col] <= 800) | df_clean[col].isna()
            invalid_count = (~valid_mask).sum()
            df_clean.loc[~valid_mask, col] = np.nan
            
            print(f"Step 3: Cleaned {col} - {invalid_count} invalid scores set to NaN")
    
    # Step 4: Clean number of test takers
    if 'Num of SAT Test Takers' in df_clean.columns:
        # Replace 's' (suppressed) with NaN
        df_clean['Num of SAT Test Takers'] = df_clean['Num of SAT Test Takers'].replace('s', np.nan)
        df_clean['Num of SAT Test Takers'] = pd.to_numeric(df_clean['Num of SAT Test Takers'], errors='coerce')
        
        # Remove unrealistic values (negative or extremely high)
        valid_mask = (df_clean['Num of SAT Test Takers'] > 0) & (df_clean['Num of SAT Test Takers'] <= 2000) | df_clean['Num of SAT Test Takers'].isna()
        invalid_count = (~valid_mask).sum()
        df_clean.loc[~valid_mask, 'Num of SAT Test Takers'] = np.nan
        print(f"Step 4: Cleaned Num of SAT Test Takers - {invalid_count} invalid values set to NaN")
    
    # Step 5: Clean percentage column
    if 'pct_students_tested' in df_clean.columns:
        # Convert percentage strings to numeric
        def clean_percentage(val):
            if pd.isna(val) or val == 'N/A':
                return np.nan
            if isinstance(val, str) and '%' in val:
                try:
                    return float(val.replace('%', '')) / 100
                except:
                    return np.nan
            try:
                return float(val)
            except:
                return np.nan
        
        df_clean['pct_students_tested'] = df_clean['pct_students_tested'].apply(clean_percentage)
        print("Step 5: Cleaned percentage column to numeric format")
    
    # Step 6: Clean academic tier rating
    if 'academic_tier_rating' in df_clean.columns:
        df_clean['academic_tier_rating'] = pd.to_numeric(df_clean['academic_tier_rating'], errors='coerce')
        # Ensure ratings are between 1-4
        valid_mask = df_clean['academic_tier_rating'].isin([1, 2, 3, 4]) | df_clean['academic_tier_rating'].isna()
        invalid_count = (~valid_mask).sum()
        df_clean.loc[~valid_mask, 'academic_tier_rating'] = np.nan
        print(f"Step 6: Cleaned academic tier rating - {invalid_count} invalid values set to NaN")
    
    # Step 7: Standardize column names
    df_clean.columns = [col.strip().replace(' ', '_').lower() for col in df_clean.columns]
    print("Step 7: Standardized column names")
    
    print(f"\n=== CLEANING COMPLETE ===")
    print(f"Final dataset shape: {df_clean.shape}")
    print(f"Rows removed: {len(df) - len(df_clean)}")
    
    return df_clean

def perform_statistical_analysis(df: pd.DataFrame) -> None:
    """
    Comprehensive statistical analysis of SAT score data
    """
    print("=== STATISTICAL ANALYSIS OF CLEANED DATA ===")
    
    # Basic statistics for numeric columns
    numeric_cols = ['num_of_sat_test_takers', 'sat_critical_reading_avg._score', 
                   'sat_math_avg._score', 'sat_writing_avg._score', 
                   'pct_students_tested', 'academic_tier_rating']
    
    # Filter existing columns
    existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    if existing_numeric_cols:
        print("\n=== DESCRIPTIVE STATISTICS ===")
        print(df[existing_numeric_cols].describe())
        
        # Missing value analysis
        print(f"\n=== MISSING VALUE ANALYSIS ===")
        missing_analysis = df[existing_numeric_cols].isnull().sum()
        for col, missing_count in missing_analysis.items():
            missing_pct = (missing_count / len(df)) * 100
            print(f"{col}: {missing_count} ({missing_pct:.1f}%)")
        
        # SAT Score Analysis
        sat_cols = [col for col in existing_numeric_cols if 'sat_' in col and 'score' in col]
        if sat_cols:
            print(f"\n=== SAT SCORE ANALYSIS ===")
            
            # Calculate total SAT scores where all sections are available
            complete_scores = df[sat_cols].dropna()
            if not complete_scores.empty:
                complete_scores['total_sat'] = complete_scores.sum(axis=1)
                
                print(f"Schools with complete SAT data: {len(complete_scores)}")
                print(f"Average Total SAT Score: {complete_scores['total_sat'].mean():.1f}")
                print(f"Standard Deviation: {complete_scores['total_sat'].std():.1f}")
                print(f"Min Total SAT: {complete_scores['total_sat'].min()}")
                print(f"Max Total SAT: {complete_scores['total_sat'].max()}")
                
                # Score distribution by percentiles
                percentiles = [25, 50, 75, 90, 95]
                print(f"\nTotal SAT Score Percentiles:")
                for p in percentiles:
                    score = np.percentile(complete_scores['total_sat'], p)
                    print(f"  {p}th percentile: {score:.0f}")
        
        # Academic tier analysis
        if 'academic_tier_rating' in df.columns:
            print(f"\n=== ACADEMIC TIER RATING DISTRIBUTION ===")
            tier_counts = df['academic_tier_rating'].value_counts().sort_index()
            for tier, count in tier_counts.items():
                pct = (count / len(df.dropna(subset=['academic_tier_rating']))) * 100
                print(f"  Tier {int(tier)}: {count} schools ({pct:.1f}%)")

def prepare_for_database(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare cleaned data for database insertion with optimal schema design
    """
    print("=== PREPARING DATA FOR DATABASE INSERTION ===")
    
    # Select and rename columns for database schema
    db_columns = {
        'dbn': 'dbn',
        'school_name': 'school_name', 
        'num_of_sat_test_takers': 'num_test_takers',
        'sat_critical_reading_avg._score': 'sat_critical_reading_avg_score',
        'sat_math_avg._score': 'sat_math_avg_score', 
        'sat_writing_avg._score': 'sat_writing_avg_score',
        'internal_school_id': 'internal_school_id',
        'contact_extension': 'contact_extension',
        'pct_students_tested': 'pct_students_tested',
        'academic_tier_rating': 'academic_tier_rating'
    }
    
    # Filter existing columns and rename
    existing_cols = {k: v for k, v in db_columns.items() if k in df.columns}
    df_db = df[list(existing_cols.keys())].copy()
    df_db = df_db.rename(columns=existing_cols)
    
    # Add calculated fields
    if all(col in df_db.columns for col in ['sat_critical_reading_avg_score', 'sat_math_avg_score', 'sat_writing_avg_score']):
        df_db['sat_total_avg_score'] = df_db[['sat_critical_reading_avg_score', 'sat_math_avg_score', 'sat_writing_avg_score']].sum(axis=1)
        # Set to NaN if any component is missing
        missing_mask = df_db[['sat_critical_reading_avg_score', 'sat_math_avg_score', 'sat_writing_avg_score']].isna().any(axis=1)
        df_db.loc[missing_mask, 'sat_total_avg_score'] = np.nan
        
        print(f"Added calculated field: sat_total_avg_score")
    
    # Add data processing metadata
    df_db['data_processed_at'] = pd.Timestamp.now()
    df_db['data_source'] = 'nyc_sat_results_2016'  # Adjust based on actual data year
    
    print(f"Final database schema columns: {list(df_db.columns)}")
    print(f"Rows prepared for insertion: {len(df_db)}")
    
    return df_db

def insert_data_to_database(df: pd.DataFrame, table_name: str = 'svitlana_sat_results') -> bool:
    """
    Insert data to PostgreSQL database with proper error handling and validation
    """
    try:
        print(f"=== INSERTING DATA TO DATABASE ===")
        print(f"Table: nyc_schools.{table_name}")
        print(f"Rows to insert: {len(df)}")
        
        # Database connection setup
        DATABASE_URL = (
            "postgresql+psycopg2://neondb_owner:npg_CeS9fJg2azZD"
            "@ep-falling-glitter-a5m0j5gk-pooler.us-east-2.aws.neon.tech:5432/neondb"
            "?sslmode=require"
        )
        
        # Create SQLAlchemy engine
        engine = create_engine(DATABASE_URL)
        
        # Insert data using pandas to_sql
        result = df.to_sql(
            name=table_name,
            con=engine,
            schema='nyc_schools',
            if_exists='replace',  # Replace existing table
            index=False,
            method='multi'  # Use efficient batch insertion
        )
        
        print(f"✅ Successfully inserted {len(df)} rows into nyc_schools.{table_name}")
        
        # Verify insertion
        with engine.connect() as connection:
            verification_query = f"SELECT COUNT(*) FROM nyc_schools.{table_name};"
            result = connection.execute(verification_query)
            count = result.fetchone()[0]
            print(f"✅ Verification: {count} rows found in database table")
            
            # Get sample of inserted data
            sample_query = f"SELECT * FROM nyc_schools.{table_name} LIMIT 3;"
            sample_result = connection.execute(sample_query)
            sample_data = sample_result.fetchall()
            print(f"\n=== SAMPLE INSERTED DATA ===")
            for i, row in enumerate(sample_data, 1):
                print(f"Row {i}: DBN={row[0]}, School={row[1][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Database insertion failed: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Provide troubleshooting suggestions
        print(f"\n=== TROUBLESHOOTING SUGGESTIONS ===")
        print("1. Check database connection")
        print("2. Verify schema permissions") 
        print("3. Check data types compatibility")
        print("4. Ensure nyc_schools schema exists")
        
        return False

def main():
    """
    Main function to execute the complete data processing pipeline
    """
    print("🚀 Starting NYC SAT Results Data Processing Pipeline")
    print("=" * 60)
    
    # Load the raw SAT results dataset
    data_path = '/Users/svitlanakovalivska/onboarding_weebet/_onboarding_data/daily_tasks/day_4/day_4_datasets/sat-results.csv'
    print(f"Loading data from: {data_path}")
    
    try:
        df_raw = pd.read_csv(data_path)
        print(f"✅ Successfully loaded {len(df_raw)} rows and {len(df_raw.columns)} columns")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Perform quality assessment
    print("\n" + "=" * 60)
    quality_report = assess_data_quality(df_raw)
    print(f"Data quality assessment completed")
    print(f"Duplicate rows found: {quality_report['duplicate_rows']}")
    
    # Clean the data
    print("\n" + "=" * 60)
    df_cleaned = comprehensive_data_cleaning(df_raw)
    
    # Perform statistical analysis
    print("\n" + "=" * 60)
    perform_statistical_analysis(df_cleaned)
    
    # Prepare data for database
    print("\n" + "=" * 60)
    df_final = prepare_for_database(df_cleaned)
    
    # Insert to database
    print("\n" + "=" * 60)
    success = insert_data_to_database(df_final)
    
    # Export cleaned data to CSV
    print("\n" + "=" * 60)
    output_path = '/Users/svitlanakovalivska/onboarding_weebet/_onboarding_data/cleaned_sat_results.csv'
    
    try:
        df_final.to_csv(output_path, index=False)
        print(f"✅ Successfully exported cleaned data to: {output_path}")
        print(f"   - Records exported: {len(df_final)}")
        print(f"   - Columns exported: {len(df_final.columns)}")
        
        # Create summary report
        print(f"\n=== DATA PROCESSING SUMMARY REPORT ===")
        print(f"Original dataset:")
        print(f"  - Rows: {len(df_raw)}")
        print(f"  - Columns: {len(df_raw.columns)}")
        print(f"  - Duplicates: {df_raw.duplicated().sum()}")
        
        print(f"\nCleaned dataset:")
        print(f"  - Rows: {len(df_final)}")
        print(f"  - Columns: {len(df_final.columns)}")
        print(f"  - Data quality improvements:")
        print(f"    * Removed duplicate rows: {len(df_raw) - len(df_cleaned)}")
        print(f"    * Standardized column names")
        print(f"    * Converted percentages to decimal format")
        print(f"    * Validated SAT score ranges (200-800)")
        print(f"    * Handled suppressed data appropriately")
        print(f"    * Added calculated total SAT scores")
        print(f"    * Added metadata columns")
        
    except Exception as e:
        print(f"❌ Failed to export cleaned data: {e}")
    
    # Final status
    print(f"\n🎯 PIPELINE COMPLETION STATUS:")
    print(f"✅ Data exploration and quality assessment completed") 
    print(f"✅ Comprehensive data cleaning applied")
    print(f"✅ Statistical analysis performed")
    if success:
        print(f"✅ Database schema designed and data inserted")
    else:
        print(f"❌ Database insertion failed")
    print(f"✅ Cleaned dataset exported to CSV")
    print(f"\n🎉 NYC SAT Results Data Processing Pipeline Complete!")

if __name__ == "__main__":
    main()