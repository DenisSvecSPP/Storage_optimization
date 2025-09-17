from energyquantified.metadata import ContractPeriod

def load_variables_test_func():

    products = {
    'TTF' : 'NL Futures Natural Gas EUR/MWh ICE-TTF OHLC',
    'DE' : 'DE Futures Power Base EUR/MWh EEX OHLC',
    }

    fro_period = {
    'MONTH' : [ContractPeriod.MONTH, 1],
    'QUARTER' : [ContractPeriod.QUARTER, 1],
    'SEASON' : [ContractPeriod.SEASON, 1],
    'YEAR' : [ContractPeriod.YEAR, 1],
    }

    return products, fro_period