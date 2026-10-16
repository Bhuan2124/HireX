"""SQLAlchemy engine, session factory and the FastAPI dependency."""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import DATABASE_URL

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    from app import models          # noqa: F401 - registers the tables
    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


def _add_missing_columns():
    """Lightweight migration: if a newer version added columns to a table that
    already exists in your database, add them, so you never have to delete
    resumex.db after updating the code."""
    from sqlalchemy import inspect, text
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            have = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name in have:
                    continue
                ddl_type = col.type.compile(dialect=engine.dialect)
                default = col.default.arg if col.default is not None and not callable(col.default.arg) else None
                clause = ""
                if isinstance(default, (int, float)) and not isinstance(default, bool):
                    clause = f" DEFAULT {default}"
                elif isinstance(default, str):
                    clause = f" DEFAULT '{default}'"
                conn.execute(text(f"ALTER TABLE {table.name} ADD COLUMN {col.name} {ddl_type}{clause}"))
                print(f"[db] added column {table.name}.{col.name}")
