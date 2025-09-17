from datetime import date

def ohlc_download_func(eq,begin, end, products, del_period, product):
        

    # print(f"begin: {begin} --- {type(begin)}")
    # print(f"del_period: {del_period} --- {type(del_period)}")
    # print(f"product: {product} --- {type(product)}")

    del_time = date(int(product[2]),del_period[product[1]][1],1)
    
    if (product[1] == 'DA'):
        
        timeseries = eq.ohlc.load(
            f'{products[product[0]]}',
            begin = begin,
            end = end,
            period = del_period[product[1]][0] # The June contract
            )
        
        
    else:
        timeseries = eq.ohlc.load(
            f'{products[product[0]]}',
            begin = begin,
            end = end,
            period = del_period[product[1]][0],
            delivery = del_time  # The June contract
            )
   
    actual = timeseries.to_dataframe()
    #print(actual)
 
    #actual.index = actual.index.tz_localize(None)
    
    return actual