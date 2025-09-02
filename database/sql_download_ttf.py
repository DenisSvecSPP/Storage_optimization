import pandas as pd
from database.db_select import db_select_func
import pyodbc



def db_select_func(conn, sql):
 
    cursor = conn.cursor()
    cursor.execute(sql)
    # Ensure data is fetched as a flat list of tuples
    rows = [tuple(row) for row in cursor.fetchall()]
    columns = [column[0] for column in cursor.description]
    cursor.close()

    return pd.DataFrame(rows, columns=columns)

def sql_download_ttf_func():
    connect_string = 'DRIVER={SQL Server};SERVER=SQLP11\SQLP11;DATABASE=Ceny_Komodit;uid=ZP_EE_komodity;pwd=Pxet_489.qweX'
    conn = pyodbc.connect(connect_string)

    sql = "SELECT date, market, contract, product_type, front, mid, delivery  FROM ICIS WHERE product_type IN ('Season', 'Year', 'Month', 'Quarter', 'Day ahead')"

    # Call db_select_func to execute the query and fetch results as a DataFrame
    result = db_select_func(conn, sql)

    conn.close()

    return result