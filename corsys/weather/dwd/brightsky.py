# -*- coding: utf-8 -*-
"""
    corsys.weather.nmm
    ~~~~~~~~~~~~~~~~~~~~~


"""
from __future__ import annotations

import json
import pytz as tz
import datetime as dt
import pandas as pd
import requests

from corsys.tools import to_date

from io import StringIO
from ...configs import Configurations
from ...system import System
from ..fcst import ScheduledForecast
from ..base import Weather


# noinspection SpellCheckingInspection
class Brightsky(ScheduledForecast):

    def __configure__(self, configs: Configurations) -> None:
        super().__configure__(configs)

        # TODO: Add sanity check
        self.address = configs.get('Brightsky', 'address', fallback='https://api.brightsky.dev/')
        self.horizon = configs.getint('Brightsky', 'horizon', fallback=5)
        if -1 > self.horizon > 10:
            raise ValueError(f"Invalid forecast horizon: {self.horizon}")

        self._variables = {
            Weather.GHI:                 'solar',
            Weather.TEMP_AIR:            'temperature',
            Weather.PRESSURE_SEA:        'pressure_msl',
            Weather.WIND_SPEED_GUST:     'wind_gust_speed',
            Weather.WIND_DIRECTION_GUST: 'wind_gust_direction'
        }

        self._variables_output = [
            Weather.GHI,
            Weather.TEMP_AIR,
            Weather.TEMP_DEW_POINT,
            Weather.HUMIDITY_REL,
            Weather.PRESSURE_SEA,
            Weather.WIND_SPEED,
            Weather.WIND_SPEED_GUST,
            Weather.WIND_DIRECTION,
            Weather.WIND_DIRECTION_GUST,
            Weather.CLOUD_COVER,
            Weather.SUNSHINE,
            Weather.VISIBILITY,
            Weather.PRECIPITATION,
            Weather.PRECIPITATION_PROB,
            'condition',
            'icon'
        ]

    def __activate__(self, system: System) -> None:
        super().__activate__(system)
        self.location = system.location

    def predict(self,
                start: dt.datetime | pd.Timestamp = pd.Timestamp.now(tz.UTC),
                end: dt.datetime | pd.Timestamp = None) -> pd.DataFrame:
        return self._request(start, end)

    # noinspection PyPackageRequirements
    def _request(self,
                 date:      pd.Timestamp,
                 date_last: pd.Timestamp = None) -> pd.DataFrame:
        """
        Submits a query to the meteoblue servers and
        converts the CSV response to a pandas DataFrame.
        
        Returns
        -------
        data : DataFrame
            column names are the weather model's variable names.
        """
        if date_last is None:
            date_last = date + dt.timedelta(days=self.horizon)
        parameters = {
            'date': date.strftime('%Y-%m-%d'),
            'last_date': date_last.strftime('%Y-%m-%d'),
            'lat': self.location.latitude,
            'lon': self.location.longitude,
            'tz': self.location.timezone.zone
        }
        response = requests.get(self.address + 'weather', params=parameters)

        if response.status_code != 200:
            raise requests.HTTPError("Response returned with error " + str(response.status_code) + ": " +
                                     response.reason)

        data = json.loads(response.text)
        data = pd.DataFrame(data['weather'])
        data['timestamp'] = pd.DatetimeIndex(pd.to_datetime(data['timestamp'], utc=True))
        data = data.set_index('timestamp').tz_convert(self.location.timezone)
        data.index.name = 'time'

        hours = pd.Series(data=data.index, index=data.index).diff().bfill().dt.total_seconds() / 3600.

        # Convert global horizontal irradiance from kWh/m^2 to W/m^2
        data['solar'] = data['solar']*hours*1000

        if data[Weather.CLOUD_COVER].isna().any():
            data[Weather.CLOUD_COVER].interpolate(method='linear', inplace=True)

        data.dropna(how='all', axis='columns', inplace=True)

        return self._rename(data)[self._variables_output]
