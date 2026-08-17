from sqlalchemy import create_engine, text
DATABASE_URL = "postgresql://rudrayadav@localhost:5432/text_to_sql"

engine = create_engine(DATABASE_URL)

def execute_sql(sql_query):
    with engine.connect() as connection:

        result = connection.execute(
            text(sql_query)
        )
        rows = result.mappings().all()

        return rows


