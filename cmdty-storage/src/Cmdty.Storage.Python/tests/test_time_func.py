# Copyright(c) 2020 Jake Fowler
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

import unittest
from cmdty_storage import time_func
from datetime import date, datetime
import pandas as pd


class TestAct365(unittest.TestCase):
    def test_daily_one_year_diff_returns_1(self):
        self.assertEqual(1.0, time_func.act_365(date(2020, 8, 5), date(2021, 8, 5)))
        self.assertEqual(1.0, time_func.act_365(datetime(2020, 8, 5), datetime(2021, 8, 5)))
        self.assertEqual(1.0, time_func.act_365(datetime(2020, 8, 5, 0, 0, 0), datetime(2021, 8, 5, 23, 45, 56)))
        self.assertEqual(1.0, time_func.act_365("2020-08-05", "2021-08-05"))
        self.assertEqual(1.0, time_func.act_365(date(2020, 8, 5), "2021-08-05"))
        self.assertEqual(1.0, time_func.act_365("2020-08-05", date(2021, 8, 5)))
        self.assertEqual(1.0, time_func.act_365(pd.Period("2020-08-05", freq='D'), pd.Period("2021-08-05", freq='D')))
        self.assertEqual(1.0, time_func.act_365(date(2020, 8, 5), pd.Period("2021-08-05", freq='D')))
        self.assertEqual(1.0, time_func.act_365(datetime(2020, 8, 5), pd.Period("2021-08-05", freq='D')))

    def test_quarterly_one_year_diff_returns_1(self):
        self.assertEqual(1.0, time_func.act_365(pd.Period(year=2020, quarter=2, freq='Q'),
                                                pd.Period(year=2021, quarter=2, freq='Q')))
        self.assertEqual(1.0, time_func.act_365(date(2020, 4, 1), pd.Period(year=2021, quarter=2, freq='Q')))
        self.assertEqual(1.0, time_func.act_365('2020-04-01', pd.Period(year=2021, quarter=2, freq='Q')))

    def test_daily_3_days_diff_returns_3_over_365(self):
        expected = 3 / 365.0
        self.assertEqual(expected, time_func.act_365(date(2020, 8, 5), date(2020, 8, 8)))
        self.assertEqual(expected, time_func.act_365(datetime(2020, 8, 5), datetime(2020, 8, 8)))
        self.assertEqual(expected, time_func.act_365(datetime(2020, 8, 5, 0, 0, 0), datetime(2020, 8, 8, 23, 45, 56)))
        self.assertEqual(expected, time_func.act_365("2020-08-05", "2020-08-08"))
        self.assertEqual(expected, time_func.act_365(date(2020, 8, 5), "2020-08-08"))
        self.assertEqual(expected, time_func.act_365("2020-08-05", date(2020, 8, 8)))
        self.assertEqual(expected, time_func.act_365(pd.Period("2020-08-05", freq='D'), pd.Period("2020-08-08", freq='D')))
        self.assertEqual(expected, time_func.act_365(date(2020, 8, 5), pd.Period("2020-08-08", freq='D')))
        self.assertEqual(expected, time_func.act_365(datetime(2020, 8, 5), pd.Period("2020-08-08", freq='D')))

    def test_quarterly_3_days_diff_returns_3_over_365(self):
        expected = 3 / 365.0
        self.assertEqual(expected, time_func.act_365(date(2021, 3, 29), pd.Period(year=2021, quarter=2, freq='Q')))
        self.assertEqual(expected, time_func.act_365('2021-03-29', pd.Period(year=2021, quarter=2, freq='Q')))


if __name__ == '__main__':
    unittest.main()
