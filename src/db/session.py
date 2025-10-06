from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor


@contextmanager
def get_db():
    return psycopg2.connect(
        host="bioseekers-db.cjg2c8w8choo.us-east-2.rds.amazonaws.com",
        port="5432",
        database="bioseekers_nasa_2025",
        user="postgres",
        password="bioseekers123",
        cursor_factory=RealDictCursor,
    )
