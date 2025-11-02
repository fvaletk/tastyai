# db/migrate.py

import os
from alembic import command
from alembic.config import Config
from dotenv import load_dotenv

# Load .env variables (useful in Docker or dev)
load_dotenv()

def run_migrations():
    alembic_cfg = Config("alembic.ini")
    
    # Optional: log config to stdout
    alembic_cfg.set_main_option("script_location", "alembic")
    
    # Set DB URL explicitly from .env
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    # Apply migrations
    command.upgrade(alembic_cfg, "head")
