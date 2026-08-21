from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv
load_dotenv
import os
DATABASE_URL = "postgresql://rudrayadav@localhost:5432/text_to_sql"
engine = create_engine(DATABASE_URL)

def generate_compact_agent_schema(target_schema="public"):
    inspector = inspect(engine)
    tables = inspector.get_table_names(schema=target_schema)
    
    schema_output = f"DATABASE SCHEMA (Schema: {target_schema})\n"
    schema_output += "=========================================\n\n"
    
    for table_name in tables:
        schema_output += f"Table: {table_name}\n"
        
        # 1. Fetch Columns cleanly: "name (type)"
        columns = inspector.get_columns(table_name, schema=target_schema)
        col_list = [f"{col['name']} ({str(col['type'])})" for col in columns]
        schema_output += f"  Columns: {', '.join(col_list)}\n"
        
        # 2. Fetch Foreign Keys so the agent knows exactly how to JOIN tables
        fkeys = inspector.get_foreign_keys(table_name, schema=target_schema)
        if fkeys:
            fk_list = []
            for fk in fkeys:
                for src_col, tgt_col in zip(fk["constrained_columns"], fk["referred_columns"]):
                    fk_list.append(f"{src_col} -> {fk['referred_table']}.{tgt_col}")
            schema_output += f"  Foreign Keys: {', '.join(fk_list)}\n"
            
        schema_output += "\n"
        
    return schema_output

print(generate_compact_agent_schema())