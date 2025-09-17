from historicl_movement_functions.ohlc_download_front import ohlc_download_front_func
from historicl_movement_functions.load_variables_test import load_variables_test_func
from energyquantified import EnergyQuantified
from datetime import datetime
import streamlit as st


@st.cache_resource(ttl=16400)
def ttf_download_hisotry_func(download_date, data_for_correlation = False):

    
    http_proxy = "http://proxy.spp.sk:8080"
    https_proxy = "http://proxy.spp.sk:8080"

    proxies = {
        "http" : http_proxy,
        "https" : https_proxy,
    }

    now = datetime.now()
    current_month_index = now.month
    if current_month_index == 12:
        current_month_index = 0

    # Initialize eq client
    eq = EnergyQuantified(api_key='08deafc1-3bc68297-bcafc06d-e92d89a6',
                        proxies=proxies)

    current_year = now.year
    products, fro_period = load_variables_test_func()
 
    dataframes = {}
    for ticker, value in products.items():


        for seanson, value in fro_period.items():
            if data_for_correlation and seanson != 'MONTH':
                continue

            product_0 = [ticker, seanson, current_year]

            df, _ = ohlc_download_front_func(eq, download_date, now.date(), products, fro_period, product_0, 1)
            df_key = f"df_{ticker}_{seanson}"

            # Store the dataframe in the dictionary with the unique key
            dataframes[df_key] = df

    return dataframes