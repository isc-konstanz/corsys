# -*- coding: utf-8 -*-
"""
    th-e-sim.preparation
    ~~~~~
    
    
"""
import logging
logger = logging.getLogger(__name__)

import os
import pytz as tz
import numpy as np
import pandas as pd
import datetime as dt
import dateutil.relativedelta as rd


def process_weather(system, meteoblue_libs='Meteoblue'):
    lat = system._configs.getfloat('Location', 'latitude')
    lon = system._configs.getfloat('Location', 'longitude')
    loc = '{0:06.2f}'.format(lat).replace('.', '') + '_' \
          + '{0:06.2f}'.format(lon).replace('.', '')

    weather = system.forecast._weather._configs
    weather_dir = os.path.join(weather.get('Database', 'dir'))
    weather_libs = os.path.join(weather.get('General', 'lib_dir'), 'weather')
    weather_lib = os.path.join(weather_libs, loc)

    if os.path.isdir(weather_dir):
        return

    os.makedirs(weather_dir, exist_ok=True)

    if not os.path.exists(weather_lib) and \
            os.path.exists(weather_libs):
        os.mkdir(weather_lib)

    logger.info('Preparing weather files for {0} in: {1}'.format(system.name, weather_dir))

    meteoblue_lib = os.path.join(meteoblue_libs, 'Locations', loc)
    if not os.path.isdir(meteoblue_libs):
        raise Exception("Unable to access meteoblue directory: {0}".format(meteoblue_libs))

    infos = []
    files = []

    for entry in os.scandir(meteoblue_libs):
        if entry.is_file() and entry.path.endswith('.csv'):
            info = pd.read_csv(entry.path, skipinitialspace=True, sep=';', header=None, index_col=[0],
                               low_memory=False).iloc[:18, :]
            info.columns = info.iloc[3]

            infos.append(info.loc[:, ~info.columns.duplicated()].dropna(axis=1, how='all'))
            files.append(pd.read_csv(entry.path, skipinitialspace=True, sep=';', header=[18], index_col=[0, 1, 2, 3, 4]))

    points = pd.concat(infos, axis=0).drop_duplicates()
    histories = pd.concat(files, axis=1)
    for point in points.columns.values:
        if abs(lat - float(points.loc['LAT', point]) > 0.001) or \
           abs(lon - float(points.loc['LON', point]) > 0.001):
            continue

        columns = [column for column in histories.columns.values if column.startswith(point + ' ')]
        history = histories[columns]
        history.columns = [c.replace(c.split(' ')[0], '').replace(c.split('[')[1], '').replace('  [', '') for c in columns]
        history['time'] = [dt.datetime(y, m, d, h, n) for y, m, d, h, n in history.index]
        history.set_index('time', inplace=True)
        history.index = history.index.tz_localize(tz.utc)
        # history.index = history.index.tz_convert('Europe/Berlin')
        history.rename(columns={' Temperature': 'temp_air', 
                                ' Wind Speed': 'wind_speed', 
                                ' Wind Direction': 'wind_direction', 
                                ' Wind Gust': 'wind_gust', 
                                ' Relative Humidity': 'humidity_rel', 
                                ' Mean Sea Level Pressure': 'pressure_sea', 
                                ' Shortwave Radiation': 'ghi', 
                                ' DNI - backwards': 'dni', 
                                ' DIF - backwards': 'dhi', 
                                ' Total Cloud Cover': 'total_clouds', 
                                ' Low Cloud Cover': 'low_clouds', 
                                ' Medium Cloud Cover': 'mid_clouds', 
                                ' High Cloud Cover': 'high_clouds', 
                                ' Total Precipitation': 'precipitation', 
                                ' Snow Fraction': 'snow_fraction'}, inplace=True)

        if os.path.isdir(meteoblue_lib):
            fill_start = history.index[-1]
            fill_end = None

            # Delete unavailable column of continuous forecasts
            del history['wind_gust']

            for file in sorted(os.listdir(meteoblue_lib)):
                path = os.path.join(meteoblue_lib, file)
                if os.path.isfile(path) and file.endswith('.csv'):
                    forecast = pd.read_csv(path, index_col='time', parse_dates=['time'])
                    forecast.columns = ['temp_air', 'wind_speed', 'wind_direction', 'humidity_rel', 'pressure_sea', 
                                        'ghi', 'dni', 'dhi', 'total_clouds', 'low_clouds', 'mid_clouds', 'high_clouds', 
                                        'precipitation', 'precipitation_convective', 'precipitation_probability', 
                                        'snow_fraction']

                    time = forecast.index[0]
                    if time.hour == 0 or time.hour == 1:
                        if fill_end is None or time < fill_end:
                            fill_end = time

                        history = forecast.loc[time:time+dt.timedelta(hours=23), history.columns].combine_first(history)

                    if os.path.exists(weather_lib):
                        forecast_file = os.path.join(weather_lib, file)
                        if not os.path.exists(forecast_file):
                            forecast = forecast.resample('1Min').interpolate(method='akima')
                            forecast[forecast[['ghi', 'dni', 'dhi']] < 0] = 0
                            forecast.to_csv(forecast_file, sep=',', encoding='utf-8') #, date_format='%Y-%m-%d %H:%M:%S')

            fill_offset = rd.relativedelta(years=10)
            fill_data = history[fill_start-fill_offset:fill_end-fill_offset]
            fill_leap = (fill_data.index.month == 2) & (fill_data.index.day == 29)
            if fill_leap.any():
                fill_data = fill_data.drop(fill_data[fill_leap].index)
            fill_data.index = fill_data.index.map(lambda dt: dt.replace(year=dt.year+10))

            history = history.combine_first(fill_data)

        # Upsample forecast to a resolution of 1 minute. Use the advanced Akima interpolator for best results
        history = history.resample('1Min').interpolate(method='akima')
        history[history[['ghi', 'dni', 'dhi']] < 0] = 0

        time = history.index[0]
        while time <= history.index[-1]:
            day = history[time:time+dt.timedelta(hours=23, minutes=59)]
            day.to_csv(os.path.join(weather_dir, time.strftime('%Y%m%d_%H%M%S') + '.csv'), sep=',', encoding='utf-8')

            if os.path.exists(weather_lib):
                loc_file = os.path.join(weather_lib, time.strftime('%Y%m%d_%H%M%S') + '.csv');
                if not os.path.exists(loc_file):
                    day.to_csv(loc_file, sep=',', encoding='utf-8')

            time = time + dt.timedelta(hours=24)


def process_system(system, opsd_libs='OPSD'):
    lat = system._configs.getfloat('Location', 'latitude')
    lon = system._configs.getfloat('Location', 'longitude')
    loc = '{0:08.4f}'.format(lat).replace('.', '') + '_' + \
          '{0:08.4f}'.format(lon).replace('.', '')

    system_dir = system._configs.get('Database', 'dir')
    system_libs = os.path.join(system._configs.get('General', 'lib_dir'), 'systems')
    system_lib = os.path.join(system_libs, loc)

    if os.path.isdir(system_dir):
        return

    os.makedirs(system_dir, exist_ok=True)

    logger.info('Preparing system files for {0} in: {1}'.format(system.name, system_dir))

    if not os.path.exists(system_lib) and \
            os.path.exists(system_libs):
        os.mkdir(system_lib)

    if not os.path.isdir(opsd_libs):
        raise Exception("Unable to access OPSD directory: {0}".format(opsd_libs))

    index = 'utc_timestamp'
    data = pd.read_csv(os.path.join(opsd_libs, 'household_data_1min.csv'),
                       skipinitialspace=True, low_memory=False, sep=',',
                       index_col=[index], parse_dates=[index])

    data.index.rename('time', inplace=True)
    data = data.filter(regex=(system.id)).dropna(how='all')
    for column in data.columns:
        column_name = column.split(system.id + '_', 1)[1] + '_energy'
        data.rename(columns={column: column_name}, inplace=True)

    columns_power = ['import_power', 'export_power']
    columns_energy = ['import_energy', 'export_energy']

    data['import_energy'] = _process_energy(data['grid_import_energy'])
    data['import_power'] = _process_power(data['grid_import_energy'])

    data['export_energy'] = _process_energy(data['grid_export_energy'])
    data['export_power'] = _process_power(data['grid_export_energy'])

    if 'pv_energy' in data.columns:
        columns_power.append('pv_power')
        columns_energy.append('pv_energy')
        data['pv_energy'] = _process_energy(data['pv_energy'])
        data['pv_power'] = _process_power(data['pv_energy'])

    columns_power.append('el_power')
    columns_energy.append('el_energy')
    data['el_energy'] = data['import_energy']
    if 'pv_energy' in data.columns:
        pv_cons = data['pv_energy'] - data['export_energy']
        data['el_energy'] += pv_cons

    data['el_power'] = _process_power(data['el_energy'])

    if 'heat_pump_energy' in data.columns:
        columns_power += ['th_power', 'hp_power']
        columns_energy += ['th_energy', 'hp_energy']

        data['hp_energy'] = _process_energy(data['heat_pump_energy'])
        data['hp_power'] = _process_power(data['heat_pump_energy'])

        # TODO: Make COP more sophisticated
        # Maybe try to differentiate between heating and warm water
        cop = 3.5
        data['th_power'] = _process_power(data['heat_pump_energy']) * cop  # , filter=False)
        
        # Offset and widening of thermal power from heat pump power, smoothen peaks and reduce offset again
        data_back = data['th_power'].iloc[::-1]
        data_back = data_back.rolling(window=200).mean()  
        data_front = data_back.rolling(window=50, win_type="gaussian", center=True).mean(std=15).iloc[::-1]
        data['th_power'] = data_front.rolling(window=150).mean().ffill().bfill()

        data['th_energy'] = 0
        for i in range(1, len(data.index)):
            index = data.index[i]
            hours = (index - data.index[i-1])/np.timedelta64(1, 'h')
            data.loc[index, 'th_energy'] = data['th_energy'][i - 1] + \
                                           data['th_power'][i] / 1000 * hours

    data = data[columns_power + columns_energy]
    time = data.index[0]
    time = time.replace(hour=0, minute=0) + dt.timedelta(hours=24)
    while time <= data.index[-1]:
        day = data[time:time+dt.timedelta(hours=23, minutes=59)]
        day.to_csv(os.path.join(system_dir, time.strftime('%Y%m%d_%H%M%S') + '.csv'), sep=',', encoding='utf-8')

        if os.path.exists(system_lib):
            day_file = os.path.join(system_lib, time.strftime('%Y%m%d_%H%M%S') + '.csv')
            if not os.path.exists(day_file):
                day.to_csv(day_file, sep=',', encoding='utf-8')

        time = time + dt.timedelta(hours=24)


def _process_energy(energy):
    energy = energy.fillna(0)
    return energy - energy[0]


def _process_power(energy, filter=True): #@ReservedAssignment
    delta_energy = energy.diff()
    delta_index = pd.Series(energy.index, index=energy.index)
    delta_index = (delta_index - delta_index.shift(1))/np.timedelta64(1, 'h')

    column_power = (delta_energy/delta_index).fillna(0)*1000

    if filter:
        from scipy import signal
        b, a = signal.butter(1, 0.25)
        column_power = signal.filtfilt(b, a, column_power, method='pad', padtype='even', padlen=15)
        column_power[column_power < 0.1] = 0

    return column_power


def process_acorn_lcl(system, acorn_libs = 'Acorn_LCL'):

    # choose categories
    key_search = _what_to_search()
    # read category statistics
    acorn_details = pd.read_csv(os.path.join(acorn_libs, 'acorn_details.csv'))#skipinitialspace=True, low_memory=False, sep=',', index_col=[index], parse_dates=[index])


    #seeking the best fitting group
    best_group = _get_best_fit_group(key_search, acorn_details)

    #get all the ids from dataset information
    grouped_blocks = _get_grouped_blocks(best_group, acorn_libs)

    #get all the data from dataset on data
    whole_energy_on_group = _get_whole_energy_data(grouped_blocks, acorn_libs)
    whole_energy_on_group = whole_energy_on_group.fillna(0)
    whole_energy_on_group = whole_energy_on_group.replace("Null", 0)

    whole_energy_on_group.to_csv(os.path.join(r"C:\Users\jw\PycharmProjects\th-e-fcst\data\acorn_processed\whole_energy_on_group.csv"), sep=',', encoding='utf-8')

    pivot_df = whole_energy_on_group.pivot(index='DateTime', columns='LCLid', values='KWh')

    #changes the stucture of the data due tue
    #datetime index and a culoum per ID    pivot_df.fillna(0)
    # zeit codierung
    pivot_df = pivot_df.replace("Null", 0)

    pivot_df.to_csv(os.path.join(r"C:\Users\jw\PycharmProjects\th-e-fcst\data\acorn_processed\pivot_df.csv"), sep=',', encoding='utf-8')


def _what_to_search():
    q_a_pair_1 = ("Household Size", "Household size : 3-4 persons")
    q_a_pair_2 = ("Structure", "Couple family with dependent children")
    q_a_pair_3 = ("House Type", "Detached house")
    q_a_pair_4 = ("House Size", "Number of Beds : 4")
    search_keys = [q_a_pair_1, q_a_pair_2, q_a_pair_3, q_a_pair_4]

    return search_keys

def _build_percentage_pd(df_category, answer):
    # df_category holds a certained slice of the arcon_group_details csv on a certain question category
    # is the answer to search for df_category = df[df['CATEGORIES'] == category]

    answer_index = 0
    for entity in df_category.REFERENCE:
        if entity == answer:
            break
        answer_index += 1

    # get_group_labels
    group_labels = [s for s in df_category.columns if "ACORN-" in s]
    #computes the probability for each "a" on q fer group
    df_cat_perc = df_category[group_labels].apply(lambda x: 100 * x / float(x.sum()))

    # return probability for the correct "a" on "q"
    df_perc_sel_answer = df_cat_perc.iloc[answer_index]

    return df_perc_sel_answer

def _get_best_fit_group(search_key,df):

    all_percetage=[]
    for _, q_a_pair in enumerate(search_key):
        # computes per q_a_pair the probability for the "a" to be right within the Acorn groups
        # and stors them in a dfs with are stored in the array all_percentage
        #q_a_pair[0]]= question; q_a_pair[0]]=answer
        all_percetage.append(_build_percentage_pd(df[df['CATEGORIES'] == q_a_pair[0]], q_a_pair[1]))
    #makes a df out of the list of dfs
    all_percetage = pd.concat(all_percetage, axis=1)
    #computes the averaged value per acron group and returns the best group
    all_percetage_mean = all_percetage.mean(axis=1)
    best_group = all_percetage_mean.idxmax()

    return best_group

def _get_grouped_blocks(best_group,acorn_libs):
    # in acron group
    # out df grouped into blocks withe the  ids to the corresponding block
    # load household informations

    df_household = pd.read_csv(os.path.join(acorn_libs, 'informations_households.csv'))
    #dataframe with all ids coresponding to acorn group
    df_household_group = df_household[(df_household['Acorn'] == best_group)]
    #new df with nutzdaten --> file indicates the blocks
    x = pd.concat([df_household_group.file, df_household_group.LCLid], axis=1)
    #grouped blocks is s df with all the ids grouped by the blocks where you find them
    grouped_blocks = x.groupby('file')

    return grouped_blocks

def _get_whole_energy_data(grouped_blocks,acorn_libs):
    # list of block to search for
    list_of_blocks = list(grouped_blocks.indices.keys())
    #appends all blocks with anergy data together
    block_energy_list = []

    for block in list_of_blocks:

        #composes the link to the data of a block
        block_link = acorn_libs + '\halfhourly_dataset\halfhourly_dataset' + block + ".csv"
        block_link = block_link.replace("halfhourly_datasetblock", "halfhourly_dataset\\block")
        df_block = pd.read_csv(os.path.join(acorn_libs, block_link), engine='python')

        # holt alle ids zum block
        arr_id_block = grouped_blocks.get_group(block).LCLid

        # collects energy data for the block
        df_block = df_block[df_block.LCLid.str.contains('|'.join(arr_id_block.array))]

        # Renaming the columns
        df_block.columns = ['LCLid', 'DateTime', 'KWh']

        # Creating date, time related columns
        df_block['DateTime'] = pd.to_datetime(df_block['DateTime'])

        # appends energy df to block energy list
        block_energy_list.append(df_block)
    #energy tor all ids and from all blocks
    whole_energy_on_group = pd.concat(block_energy_list)

    return whole_energy_on_group

def _prepair_time_triangular(whole_energy_on_group):
    timestamp_s = whole_energy_on_group.DateTime.map(pd.Timestamp.timestamp)

    day = 24 * 60 * 60
    year = (365.2425) * day

    whole_energy_on_group['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    whole_energy_on_group['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    whole_energy_on_group['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    whole_energy_on_group['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    return whole_energy_on_group