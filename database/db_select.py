# import pandas as pd
import logging
# import pyodbc

def db_select_func(conn, sql):
    result = None  # Initialize result to None
    try:
        cursor = conn.cursor()
        result = cursor.execute(sql).fetchall()
        cursor.close()
    except Exception as e:
        logging.critical("Unexpected error occurred when selecting date from DB ", exc_info=True)
        
    return result if result is not None else []  # Ensure `result` is an empty list if an error occurred

