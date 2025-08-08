#!/usr/bin/env python3
"""
Test script for PostgreSQL connection and data saving verification
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime

# Database configuration (same as in notebook)
DATABASE_URL = (
    "postgresql+psycopg2://neondb_owner:npg_CeS9fJg2azZD"
    "@ep-falling-glitter-a5m0j5gk-pooler.us-east-2.aws.neon.tech:5432/neondb"
    "?sslmode=require"
)

def test_database_connection():
    """Test database connection"""
    print("=== DATABASE CONNECTION TEST ===")
    
    try:
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as connection:
            # Test connection
            result = connection.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"✅ Connection successful!")
            print(f"PostgreSQL: {version[:80]}...")
            
            # Check nyc_schools schema
            result = connection.execute(text("""
                SELECT schema_name FROM information_schema.schemata 
                WHERE schema_name = 'nyc_schools'
            """))
            schema_exists = result.fetchone()
            
            if schema_exists:
                print("✅ Schema nyc_schools exists")
            else:
                print("❌ Schema nyc_schools DOES NOT EXIST!")
                return False
            
            # Check Svitlana tables
            result = connection.execute(text("""
                SELECT table_name
                FROM information_schema.tables 
                WHERE table_schema = 'nyc_schools' 
                AND table_name LIKE 'svitlana%'
                ORDER BY table_name;
            """))
            
            tables = result.fetchall()
            print(f"\n=== SVITLANA TABLES IN DATABASE ===")
            for table in tables:
                table_name = table[0]
                
                # Count records
                count_result = connection.execute(text(f"SELECT COUNT(*) FROM nyc_schools.{table_name}"))
                count = count_result.fetchone()[0]
                print(f"  - {table_name}: {count} records")
            
            return True
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def test_data_insertion():
    """Test saving test data"""
    print("\n=== DATA INSERTION TEST ===")
    
    try:
        engine = create_engine(DATABASE_URL)
        
        # Create test data
        test_data = pd.DataFrame({
            'dbn': ['TEST001', 'TEST002', 'TEST003'],
            'school_name': ['Test School 1', 'Test School 2', 'Test School 3'],
            'num_test_takers': [50, 75, 100],
            'sat_critical_reading_avg_score': [450, 500, 550],
            'sat_math_avg_score': [460, 510, 560],
            'sat_writing_avg_score': [440, 490, 540],
            'sat_total_avg_score': [1350, 1500, 1650],
            'internal_school_id': [999001, 999002, 999003],
            'data_processed_at': [datetime.now()] * 3,
            'data_source': ['test_script'] * 3
        })
        
        # Test table name
        test_table_name = 'svitlana_test_connection'
        
        print(f"Attempting to save {len(test_data)} test records to {test_table_name}...")
        
        # Save to database
        result = test_data.to_sql(
            name=test_table_name,
            con=engine,
            schema='nyc_schools',
            if_exists='replace',
            index=False,
            method='multi'
        )
        
        print(f"✅ Test data saved successfully!")
        
        # Verify saving
        with engine.connect() as connection:
            verification_query = text(f"SELECT COUNT(*) FROM nyc_schools.{test_table_name}")
            result = connection.execute(verification_query)
            count = result.fetchone()[0]
            print(f"✅ Verification: {count} records found in database")
            
            # Show sample data
            sample_query = text(f"SELECT * FROM nyc_schools.{test_table_name} LIMIT 2")
            sample_result = connection.execute(sample_query)
            sample_data = sample_result.fetchall()
            
            print(f"\n=== SAMPLE SAVED DATA ===")
            for i, row in enumerate(sample_data, 1):
                print(f"Record {i}: DBN={row[0]}, School={row[1]}, SAT_Total={row[6]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Saving error: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

def check_notebook_issue():
    """Check potential notebook issue"""
    print("\n=== NOTEBOOK ISSUE DIAGNOSIS ===")
    
    try:
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as connection:
            # Check specific notebook tables
            target_tables = [
                'svitlana_experement_sat_result',
                'svitlana_experement_sat_results'
            ]
            
            for table_name in target_tables:
                try:
                    count_result = connection.execute(text(f"SELECT COUNT(*) FROM nyc_schools.{table_name}"))
                    count = count_result.fetchone()[0]
                    
                    # Get last update time
                    time_result = connection.execute(text(f"""
                        SELECT MAX(data_processed_at) 
                        FROM nyc_schools.{table_name} 
                        WHERE data_processed_at IS NOT NULL
                    """))
                    last_update = time_result.fetchone()[0]
                    
                    print(f"✅ {table_name}:")
                    print(f"    Records: {count}")
                    print(f"    Last update: {last_update}")
                    
                except Exception as e:
                    print(f"❌ {table_name}: NOT FOUND ({str(e)[:50]}...)")
            
            # Check access rights
            print(f"\n=== ACCESS RIGHTS CHECK ===")
            try:
                connection.execute(text("SELECT current_user, session_user"))
                result = connection.execute(text("SELECT current_user, session_user"))
                user_info = result.fetchone()
                print(f"Current user: {user_info[0]}")
                print(f"Session user: {user_info[1]}")
                
                # Check schema rights
                schema_rights = connection.execute(text("""
                    SELECT privilege_type 
                    FROM information_schema.schema_privileges 
                    WHERE schema_name = 'nyc_schools' 
                    AND grantee = current_user
                """))
                
                rights = [row[0] for row in schema_rights.fetchall()]
                print(f"Rights on nyc_schools schema: {', '.join(rights) if rights else 'NO RIGHTS'}")
                
            except Exception as e:
                print(f"Cannot check rights: {e}")
        
    except Exception as e:
        print(f"❌ Diagnosis error: {e}")

if __name__ == "__main__":
    print("POSTGRESQL CONNECTION DIAGNOSIS")
    print("=" * 50)
    
    # Test 1: Connection
    connection_ok = test_database_connection()
    
    # Test 2: Data insertion
    if connection_ok:
        insertion_ok = test_data_insertion()
    
    # Test 3: Issue diagnosis
    check_notebook_issue()
    
    print("\n" + "=" * 50)
    print("DIAGNOSIS COMPLETED")