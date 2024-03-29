# -*- coding: utf-8 -*-
"""
    corsys.weather.fcst
    ~~~~~~~~~~~~~~~~~~~
    
    This module provides the :class:`corsys.weather.WeatherForecast`, used as reference to
    calculate e.g. photovoltaic installations generated power. The provided environmental data
    contains temperatures and horizontal solar irradiation, which can be used, to calculate the
    effective irradiance on defined, tilted photovoltaic systems.
    
"""
from __future__ import annotations
from abc import abstractmethod

import pytz as tz
import datetime as dt
import pandas as pd

from ..tools import to_date, floor_date, ceil_date
from ..configs import Configurations
from .base import Weather, WeatherException
from .db import DatabaseWeather


class WeatherForecast(Weather):

    def get(self,
            start: pd.Timestamp | dt.datetime = dt.datetime.now(),
            end:   pd.Timestamp | dt.datetime = None,
            **kwargs) -> pd.DataFrame:
        """ 
        Retrieves the forecasted data for a specified time interval

        :param start: 
            the start time for which forecasted data will be looked up for.
            For many applications, passing datetime.datetime.now() will suffice.
        :type start: 
            :class:`pandas.Timestamp` or datetime
        
        :param end: 
            the end time for which forecasted data will be looked up for.
            For many applications, passing datetime.datetime.now() will suffice.
        :type end: 
            :class:`pandas.Timestamp` or datetime
        
        :returns: 
            the forecasted data, indexed in a specific time interval.
        
        :rtype: 
            :class:`pandas.DataFrame`
        """
        return self._get_range(self.predict(start, end, **kwargs), start, end)

    # noinspection PyMethodMayBeStatic
    def _get_range(self,
                   forecast: pd.DataFrame,
                   start:    pd.Timestamp | dt.datetime,
                   end:      pd.Timestamp | dt.datetime) -> pd.DataFrame:
        if start is None or start < forecast.index[0]:
            start = forecast.index[0]
        if end is None or end > forecast.index[-1]:
            end = forecast.index[-1]
        return forecast[(forecast.index >= start) & (forecast.index <= end)]

    @abstractmethod
    def predict(self, *args, **kwargs) -> pd.DataFrame:
        pass


# noinspection PyAbstractClass
class ScheduledForecast(WeatherForecast, DatabaseWeather):

    def __configure__(self, configs: Configurations) -> None:
        super().__configure__(configs)
        self.interval = configs.getint('General', 'interval', fallback=60)
        self.delay = configs.getint('General', 'delay', fallback=0)

    def get(self,
            start: pd.Timestamp | dt.datetime = dt.datetime.now(tz.UTC),
            end:   pd.Timestamp | dt.datetime = None,
            **kwargs) -> pd.DataFrame:

        # Calculate the available forecast start and end times
        timezone = self.system.location.timezone
        end = to_date(end, timezone=timezone)
        start = to_date(start, timezone=timezone)
        start_schedule = floor_date(start, self.database.timezone, f"{self.interval}T")
        start_schedule += dt.timedelta(minutes=self.delay)
        if start_schedule > start:
            start_schedule -= dt.timedelta(minutes=self.interval)

        if self.database.exists(start_schedule):
            forecast = self.database.read(start_schedule).tz_convert(timezone)

        elif start < pd.Timestamp.now(timezone):
            raise WeatherException("Unable to read persisted historic forecast")

        else:
            forecast = self.predict(start_schedule, **kwargs)

            self.database.write(forecast, start=start_schedule)

        return self._get_range(forecast, start_schedule, end)
