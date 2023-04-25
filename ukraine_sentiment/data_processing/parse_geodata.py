import pandas as pd
import geopandas
import os
import re
from bs4 import BeautifulSoup
import numpy as np


def clean_coords(s):
    s = re.sub(' ', '', s)
    s = s.split('\n')
    out = []
    for val in s:
        if len(val) == 0:
            continue
        val = [float(x) for x in val.split(',')]
        out.append(val)
    return out


def process_KML(fpath):
    with open(fpath, 'r') as file:
        raw_data = file.read()

    soup = BeautifulSoup(raw_data, features='xml')
    data = []
    for placemark in soup.find_all('Placemark'):
        name = placemark.find('name')
        coords = placemark.find('coordinates')
        description = placemark.find('description')

        data.append({
            'name': name.get_text() if name else np.nan,
            'coords': clean_coords(coords.get_text()) if coords else np.nan,
            'twitter_source': description.get_text() if description else np.nan
        })

    df = pd.DataFrame(data)
    df = df.dropna().drop_duplicates(subset=['name', 'twitter_source'])

    return df


def parse_KML_files(kml_files, data_path):
    df = None

    for kml_file in kml_files:
        fpath = os.path.join(data_path, kml_file)
        file_df = process_KML(fpath)

        df = file_df if df is None else pd.concat([df, file_df], ignore_index=True)

    return df


def process_name(row):
    out = {'date': np.nan,
           'date_str': np.nan,
           'source': np.nan,
           'media_type': np.nan,
           'twitter_url': np.nan,
           'geo_url': np.nan,
           'event_description': np.nan}

    month, day, year, time = [None] * 4

    if not isinstance(row['name'], str):
        return out

    data = row['name'].split('-')
    if len(data) > 4:
        datestr, source, media, *comment = data
        data = [datestr, source, media, '-'.join(comment)]

    if len(data) == 4:
        datestr, source, media, comment = data
        date_data = datestr.split()
        if len(date_data) == 3:
            day, time, month = date_data
            year = 2022
        elif len(date_data) == 4:
            day, time, month, year = date_data
        date = f'{month} {day}, {year} {time}'
        out['date'] = pd.to_datetime(date)
        out['date_str'] = date
        out['source'] = source
        out['media_type'] = media
        out['event_description'] = comment

    urls = row['twitter_source']
    if not pd.isna(urls):
        urls = [x.replace('<br>', '').strip() for x in urls.split('Geo:')]
        if len(urls) == 2:
            out['twitter_url'] = urls[0] if len(urls[0]) > 0 and 'https:' in urls[0] else np.nan
            out['geo_url'] = urls[1] if len(urls[1]) > 0 and 'https:' in urls[0] else np.nan
        else:
            out['twitter_url'] = urls[0] if len(urls[0]) > 0 and 'https:' in urls[0] else np.nan

    return pd.Series(out)


def kml_data_to_geodata(kml_df, geometry_col='coords', crs='EPSG:4326'):
    event_df = kml_df.join(kml_df.apply(process_name, axis=1)).dropna()

    # KML coords are lon,lat, and geopandas expects lon, lat
    event_df = event_df.join((event_df[geometry_col].str[0]
                              .apply(lambda x: pd.Series({'lon': x[0], 'lat': x[1]}))
                              .assign(geometry=lambda x: geopandas.points_from_xy(x=x['lon'], y=x['lat']))))
    event_df = geopandas.GeoDataFrame(event_df, crs=crs)
    return event_df


def match_kml_locations_to_shapefile_regions(kml_df, geo_df):
    data = []
    for idx, row in kml_df.iterrows():
        point = row.geometry
        region_mask = geo_df.contains(point)
        in_ukraine = region_mask.any()
        if not in_ukraine:
            continue
        region = geo_df.loc[region_mask]
        if region.shape[0] == 1:
            region_data = {'index': idx,
                           'country': region.ISO.values[0],
                           'name_1': region.NAME_1.values[0],
                           'name_2': region.NAME_2.values[0],
                           'varname': region.VARNAME_2.values[0]}
            data.append(region_data)
        elif region.shape[0] > 1:
            raise ValueError('Found a single geographic Point associated with multiple regions:\n'
                             f'Point: {point}\n'
                             f'Point row: {idx}\n'
                             f'Regions: {region.values}\n')
    region_df = pd.DataFrame(data).set_index('index')
    final_df = kml_df.join(region_df, how='right').sort_index().reset_index(drop=True)

    return final_df


def save_processed_shapefile(df, shapefile_path, out_fname):
    processed_shapefile_path = os.path.join(shapefile_path, 'Processed')
    if not os.path.isdir(processed_shapefile_path):
        os.mkdir(processed_shapefile_path)
    out_path = os.path.join(processed_shapefile_path, out_fname)

    # Shapefiles don't support datetime object, convert to string
    df['date'] = df.date.dt.strftime('%m-%d-%y %H:%M')
    df.drop(columns=['coords'], inplace=True)
    df.to_file(out_path)


def find_processed_fpath(shapefile_path):
    processed_dir = os.path.join(shapefile_path, 'Processed')
    files = os.listdir(processed_dir)
    fname = [x for x in files if x.endswith('shp')][0]

    fpath = os.path.join(processed_dir, fname)
    return fpath


def fetch_and_load_processed_geodata(shapefile_path):
    fpath = find_processed_fpath(shapefile_path)
    df = geopandas.read_file(fpath)

    df['date'] = df.date.apply(pd.to_datetime)

    # un-truncate column names
    df.rename(columns={'twitter_so': 'twitter_source',
                       'event_desc': 'event_description',
                       'twitter_ur': 'twitter_url'},
              inplace=True)

    return df


def process_raw_geodata(kml_files, shapefile, data_path, shapefile_path,
                        return_df=False,
                        out_fname='processed_data.shp',
                        save_to_file=True):
    df = parse_KML_files(kml_files, data_path)
    df = kml_data_to_geodata(df)

    shapefile_fpath = os.path.join(shapefile_path, shapefile)
    ukraine = geopandas.read_file(shapefile_fpath)
    df = match_kml_locations_to_shapefile_regions(df, ukraine)

    if save_to_file:
        save_processed_shapefile(df, shapefile_path, out_fname)
    if return_df:
        return df
