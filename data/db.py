#data/db.py
# this file serves to abstract away the database connection and operations

import subprocess
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
import os
from dotenv import load_dotenv
from data.models import Base

# Load environment variables from .env file
load_dotenv()

# Get database connection parameters from environment variables
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DATABASE_TYPE = os.getenv("DATABASE_TYPE", "postgresql")
DB_NAME = os.getenv("DB_NAME")

# Construct the database URL
DB_URL = f"{DATABASE_TYPE}://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Singleton pattern for database engine
_engine = None
_Session = None

def init_db():
    """Initialize the database, creating tables if they don't exist."""
    global _engine, _Session
    
    if _engine is None:
        print(f"Connecting to database: {DB_HOST}:{DB_PORT}/{DB_NAME}")  # Changed to DB_NAME
        _engine = create_engine(DB_URL, poolclass=NullPool)
        _Session = sessionmaker(bind=_engine)
        Base.metadata.create_all(_engine)
    
    return _engine

def get_db_session() -> Session:
    """Get a new database session."""
    global _engine, _Session
    
    if _engine is None:
        init_db()
    
    return _Session()

def close_db_session(session: Session):
    """Close the database session."""
    if session:
        session.close()

def backup_postgresql_db(backup_dir=None, pg_dump_executable=None):
    """
    Backs up the PostgreSQL database to a specified directory.
    If no directory is specified, it creates a backup directory in the project root.
    """
    try:
        # Determine backup directory
        if backup_dir is None:
            backup_dir = os.path.join(os.getcwd(), 'db_backups')  # Default to project root
        
        # Ensure the backup directory exists
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create a timestamp for the backup file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(backup_dir, f"{DB_NAME}_{timestamp}.dump")
        if not pg_dump_executable:
            pg_dump_executable = "pg_dump"
        
        # Construct the pg_dump command
        pg_dump_cmd = [
            pg_dump_executable,
            "-h", DB_HOST,
            "-p", DB_PORT,
            "-U", DB_USERNAME,
            "-d", DB_NAME,
            "-f", backup_file
        ]
        
        # Execute the pg_dump command
        print(f"Backing up database to: {backup_file}")
        
        # Need to pass the password via environment variable
        env = os.environ.copy()
        env["PGPASSWORD"] = DB_PASSWORD
        
        process = subprocess.Popen(pg_dump_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            print(f"Database backup successful: {backup_file}")
        else:
            print(f"Database backup failed. Error: {stderr.decode()}")
        
    except Exception as e:
        print(f"Error during database backup: {e}")