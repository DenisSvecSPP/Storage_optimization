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

from cmdty_storage import utils
import dateutil.parser as dt_parser
from datetime import date
import pandas as pd


def act_365(start: utils.ForwardPointType, end: utils.ForwardPointType) -> float:
    start = _to_date(start)
    end = _to_date(end)
    return (end - start).days / 365


def _to_date(date_like: utils.ForwardPointType) -> date:
    if isinstance(date_like, str):
        date_like = dt_parser.parse(date_like)
    else:
        if isinstance(date_like, pd.Period):
            date_like = date_like.asfreq('D', 's')
    return date(date_like.year, date_like.month, date_like.day)