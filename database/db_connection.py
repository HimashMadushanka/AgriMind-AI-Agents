import os
from pathlib import Path
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = f"sqlite:///{ROOT / 'paddy_assistant.db'}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_URL)
engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)

# Use `engine.begin()` for transactions. Before deployment, add SQLAlchemy models,
# access controls, retention/deletion policies, and avoid logging sensitive farmer data.
