from datetime import date
from energyquantified.metadata import ContractPeriod
from dateutil.relativedelta import relativedelta
import pandas as pd


def ohlc_download_front_func(_eq, begin, end, products, del_period, product, front):
   
    del_time = date(int(product[2]),del_period[product[1]][1],1)

    timeseries = _eq.ohlc.load(
            f'{products[product[0]]}',
            begin = begin,
            end = end,
            period = del_period[product[1]][0],
            front=front
        )
 
    actual = timeseries.to_dataframe()

    if del_period[product[1]][0] == ContractPeriod.MONTH :
        to_time0 = del_time + relativedelta(months=1, hours = -1)
        mwh = len(pd.date_range(start=del_time, end=to_time0, freq='h'))

    elif del_period[product[1]][0] == ContractPeriod.QUARTER :
        to_time0 = del_time + relativedelta(months=3, hours = -1)

        mwh = len(pd.date_range(start=del_time, end=to_time0, freq='h'))

    elif del_period[product[1]][0] == ContractPeriod.SEASON :
        to_time0 = del_time + relativedelta(months=6, hours = -1)

        mwh = len(pd.date_range(start=del_time, end=to_time0, freq='h'))

    elif del_period[product[1]][0] == ContractPeriod.YEAR :
        to_time0 = del_time + relativedelta(months=12, hours = -1)
        mwh = len(pd.date_range(start=del_time, end=to_time0, freq='h'))
    
    return actual, mwh