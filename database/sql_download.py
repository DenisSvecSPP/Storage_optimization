import pandas as pd
from database.db_select import db_select_func
import pyodbc
from concurrent.futures import ThreadPoolExecutor, as_completed


def sql_download_func(start_of_year, today, indicator_selected):

    connect_string = 'DRIVER={SQL Server};SERVER=SQLP11\SQLP11;DATABASE=ENTSOG_ZP;uid=ZP_EE_komodity;pwd=Pxet_489.qweX'

    def new_db_connection():
        return pyodbc.connect(connect_string)

    def run_query(sql, data_dict_func, conn):
        try:
            result = db_select_func(conn, sql)
           
            if result:
                data_dicts = data_dict_func(result)
              
                return pd.DataFrame(data_dicts)
            else:
               
                return pd.DataFrame()
        except Exception as e:
      
            return pd.DataFrame()  # Return an empty DataFrame on error

    # Define the SQL queries and their corresponding data dict functions
    sql_queries = {
        "entsog_points": (f"""SELECT date, country, point_label, point_type, direction_key, indicator, SUM(value) AS value 
                              FROM entsog_points
                              WHERE indicator LIKE '{indicator_selected}'  AND point_type != 'N/A' AND date >= '{start_of_year}' AND date <= '{today}'
                              GROUP BY date, country, point_label, point_type, direction_key, indicator 
                              ORDER BY date""",
                          lambda result: [{'date': row.date, 'country': row.country, 'point_label': row.point_label, 'point_type': row.point_type, 
                                           'direction_key': row.direction_key, 'indicator': row.indicator, 'value': row.value} 
                                          for row in result]),
        
        "agsi_storage": (f"""SELECT date, name AS country, gasInStorage, full_per, injection, withdrawal, workingGasVolume 
                             FROM agsi_storage
                             WHERE date >= '{start_of_year}' AND date <= '{today}'
                             ORDER BY date""",
                         lambda result: [{'date': row.date, 'country': row.country, 'gasInStorage': row.gasInStorage, 'full_per': row.full_per, 
                                          'injection': row.injection, 'withdrawal': row.withdrawal, 'workingGasVolume': row.workingGasVolume} 
                                         for row in result]),

        "alsi_lng": (f"""SELECT date, name AS country, inventoryGWh AS inventory, sendOut, dtmiGWh AS dtmi 
                           FROM alsi_lng 
                           WHERE date >= '{start_of_year}' AND date <= '{today}'
                           ORDER BY date""",
                     lambda result: [{'date': row.date, 'country': row.country, 'inventory': row.inventory, 'sendOut': row.sendOut, 
                                      'dtmi': row.dtmi} 
                                     for row in result])
    }

    dfs = {}

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(run_query, sql, data_dict_func, new_db_connection()): name for name, (sql, data_dict_func) in sql_queries.items()}
        
        for future in as_completed(futures):
            name = futures[future]
            dfs[name] = future.result()

    # dfs["entsog_points"]['country'] = dfs["entsog_points"]['country'].astype('category')
    # dfs["entsog_points"]['point_type'] = dfs["entsog_points"]['point_type'].astype('category')
    # dfs["entsog_points"]['direction_key'] = dfs["entsog_points"]['direction_key'].astype('category')
    # dfs["entsog_points"]['indicator'] = dfs["entsog_points"]['indicator'].astype('category')

    dfs["alsi_lng"]['sendIn'] = dfs["alsi_lng"].groupby('country')['inventory'].diff().clip(lower=0)
    dfs["alsi_lng"]['value_y'] = dfs["alsi_lng"]['sendIn'] + dfs["alsi_lng"]['sendOut'].where(dfs["alsi_lng"]['sendIn'] > 0, 0)

    return dfs["entsog_points"], dfs["agsi_storage"], dfs["alsi_lng"]


#---------------------------------------------------------------------------------------------------------------------------------------
# import pandas as pd
# from gas_board_entsog_gie.database.db_select import db_select_func
# import pyodbc
# import streamlit as st


# @st.cache_resource
# def sql_download_func(start_of_year, today):


#     http_proxy = "http://proxy.spp.sk:8080"
#     https_proxy = "http://proxy.spp.sk:8080"

#     proxies = {
#         "http" : http_proxy,
#         "https" : https_proxy,
#     }

#     connect_string = 'DRIVER={SQL Server};SERVER=SQLP11\SQLP11;DATABASE=ENTSOG_ZP;uid=ZP_EE_komodity;pwd=Pxet_489.qweX'


#     conn = pyodbc.connect(connect_string)
#     sql = f"""SELECT date, country, point_label, point_type, direction_key, indicator, SUM(value)
#         AS value FROM entsog_points
#         WHERE 
#         date >= '{start_of_year}' AND 
#         date <= '{today}'
#         GROUP BY date, country, point_label, point_type, direction_key, indicator ORDER BY date"""
    
#     result = db_select_func(conn, sql)
    
#     data_dicts = [{'date': row.date, 'country': row.country, 'point_label': row.point_label, 'point_type': row.point_type,
#             'direction_key': row.direction_key, 'indicator': row.indicator, 'value': row.value} 
#             for row in result]
#     df = pd.DataFrame(data_dicts)

#     sql = f"""SELECT date, name, gasInStorage, full_per, injection, withdrawal FROM agsi_storage
#         WHERE 
#         date >= '{start_of_year}' AND 
#         date <= '{today}'
#         ORDER BY date"""
    
#     conn = pyodbc.connect(connect_string)
#     result = db_select_func(conn, sql)
#     storage_data_dicts = [{'date': row.date, 'country': row.name, 'gasInStorage': row.gasInStorage, 'full_per': row.full_per,
#             'injection': row.injection, 'withdrawal': row.withdrawal} 
#             for row in result]

#     # Convert to DataFrame
#     agsi_storage_df = pd.DataFrame(storage_data_dicts)

#     sql = f"""SELECT date, name, inventoryGWh, sendOut, dtmiGWh  FROM alsi_lng 
#         WHERE 
#         date >= '{start_of_year}' AND 
#         date <= '{today}'
#         ORDER BY date"""
    
#     conn = pyodbc.connect(connect_string)
#     result = db_select_func(conn, sql)
#     lng_data_dicts = [{'date': row.date, 'country': row.name, 'inventory': row.inventoryGWh, 'sendOut': row.sendOut,
#             'dtmi': row.dtmiGWh} 
#             for row in result]
    
#     alsi_lng_df = pd.DataFrame(lng_data_dicts)


#     conn.close()

#     return df, agsi_storage_df, alsi_lng_df, start_of_year