from sqlalchemy import create_engine, text
from dotenv import load_dotenv
load_dotenv()
import os
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)

def execute_sql(sql_query):
    with engine.connect() as connection:

        result = connection.execute(
            text(sql_query)
        )
        rows = result.mappings().all()

        return rows




