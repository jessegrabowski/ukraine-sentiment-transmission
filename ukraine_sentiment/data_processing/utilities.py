import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from zipfile import ZipFile
import geopandas
from shapely.geometry import LineString, Polygon
from fiona.drvsupport import supported_drivers

from itertools import combinations

import pytensor
import pytensor.tensor as pt

supported_drivers['KML'] = 'rw'
supported_drivers['KMZ'] = 'rw'

event_labels = {'000': 'Biological Attack', 
                '001': 'Artillery (Operational)', 
                '002': 'Artillery (Destroyed)',
                '010': 'Bombing/Shelling', 
                '011': 'Transport Infrastructure (Operational)', 
                '012': 'Transport Infrastructure (Destroyed)',
                '020': 'Pictures',
                '021': 'Helicopter (Operational)',
                '022': 'Helicopter (Destroyed)',
                '030': 'Video',
                '031': 'Fighter/Bomber (Operational)',
                '032': 'Fighter/Bomber (Destroyed)',
                '040': 'Nuclear / Radiological Hazard',
                '041': 'Tank (Operational)',
                '042': 'Tank (Destroyed)',
                '050': 'Funerals/Deaths/Mass Grave',
                '051': 'Ship (Operational)',
                '052': 'Ship (Destroyed)',
                '060': 'Chemical attack',
                '061': 'Vehicles (Operational)',
                '062': 'Vehicles (Destroyed)',
                '070': 'Other Losses',
                '071': 'Launcher (Operational)',
                '080': 'Nuclear Powerplant Enerhodar',
                '081': 'Transport plane',
                '090': 'Fighting',
                '091': 'Trenches',
                '092': 'Troops (Destroyed)',
                '100':'Troop Movement',
                '101':'POWs',
                '110':'Drone (Operational)',
                '111':'UAV/Cruise/Ballistic Missile',
                '120':'Trenches',
                '121':'Airfield',
                '131': 'Camp/Base',
                '141': 'Patrol Boat (Operational)',
                '151': 'Submarine (Operational)',
                '161': 'Movement by train',
                '171': 'Armored Vehicle (Operational)',
                '172': 'Armored Vehicle (Destroyed)',
                '192': 'Damaged Buildings',
                '193': 'Annhilated City',
                '201': 'MLRS',
                'FLAG': 'Takeover'}

from functools import reduce 

def get_slice(i, n_lags):
    return slice(n_lags - (i+1), -(i+1))

def make_lag_matrix(X, n_lags):
    lag_idxs = reversed(range(n_lags))
    return np.concatenate([X[get_slice(i, n_lags)] for i in lag_idxs], axis=-1)

def make_lag_df(data, n_lags):
    df = reduce(lambda left, right: pd.concat([left, right], axis=1), [pd.DataFrame(data).shift(i) for i in range(1, n_lags+1)]).dropna()
    return df

def compute_mu(X, beta, n_eqs):    
    mu = pt.concatenate([X @ beta[i].ravel()[:, None] for i in range(n_eqs)], axis=-1)

    return mu

@np.vectorize
def str_add(x, y):
    return f'{x}{y}'

def make_coords_and_indices(n, T, region_names):
    if region_names is None:
        ALPHA = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        regions = ALPHA[:n]
    else:
        regions = region_names
    
    cov_mat_names = str_add(np.array(regions), np.array(regions)[:, None]).T
    
    pairs = list(combinations(regions, 2)) + list(combinations(regions[::-1], 2))[::-1]
    region_dict = {k:v for v, k in enumerate(regions)}
    pair_idx = np.array([region_dict[x[0]] for x in pairs])

    pairs = ['_'.join(x) for x in pairs]

    tril_idx = np.tril_indices(len(regions), k=-1)
    triu_idx = np.triu_indices(len(regions), k=1)

    n_regions = len(regions)
    triang_size = (n_regions * (n_regions - 1)) // 2

    coords = {
        'region_1':regions,
        'region_2':regions,
        'region_pairs':pairs,
        'position':['diag', 'offdiag'],
        'time':np.arange(T),
        'triang_names':cov_mat_names[np.tril_indices(len(regions))].ravel()
    }
    
    return coords, pair_idx, tril_idx, triu_idx, triang_size

GEOTWEET_URL_PATTERN = '({}{})'.format(re.escape("https://twitter.com/GeoConfirmed/status/"), '\d{19}')
SOURCE_URL_PATTERN = '({}\w+/status/{})'.format(re.escape("https://twitter.com/"), '\d{19}')

def get_urls(s, pattern, name):
    res = {f'{name}_0':np.nan}
    urls = re.findall(pattern, s)
    for i, url in enumerate(urls):
        res[f'{name}_{i}'] = url
    return res

def erase_urls(s, pattern):
    return re.sub(pattern, '', s)

def erase_tags(s):
    return re.sub('(<br>[A-Z]\w{2}: <br>)', '', s)

def extract_date(s):
    date = re.search('\d{2} \d{4} [A-Z]{3} \d{4}?', s)
    return date[0] if date is not None else np.nan

TAGS = ['VID', 'UAV', 'PIC', 'SAT', 'DRONE']
def get_tags(s):
    res = dict.fromkeys(TAGS, False)
    tags = re.findall('(VID)|(UAV)|(PIC)|(SAT)|(DRONE)', s)
    for match in tags:
        for tag in match:
            if len(tag) > 0:
                res[tag.strip()] = True
    return res

def clean_name(s):
    no_date = re.sub('\d{2} \d{4} [A-Z]{3} \d{4}?', '', s)
    no_tags = re.sub('(VID)|(UAV)|(PIC)|(SAT)|(DRONE)', '', no_date)
    tokens = no_tags.split('-')
    return '-'.join([token.strip() for token in tokens if len(token.strip()) > 0])

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


def load_geoconfirmed_data():
    # Load KML data
    with ZipFile('data/geoconfirmed/@GeoConfirmed - War Ukraine - All datapoints.kmz', 'r') as kmz:
        kml = kmz.open(kmz.filelist[0].filename, 'r').read()
    soup = BeautifulSoup(kml, 'xml')    
    data = []
    for placemark in soup.find_all('Placemark'):
        name = placemark.find('name')
        coords = placemark.find('coordinates')
        description = placemark.find('description')
        icon = placemark.find('styleUrl')
        data.append({
            'name':name.get_text() if name else np.nan,
            'coords':clean_coords(coords.get_text()) if coords else np.nan,
            'icon':icon.get_text(),
            'twitter_source':description.get_text() if description else np.nan
        })

    # KML to dataframe
    df = pd.DataFrame(data)

    # Get mapping from the icon .png image name to something human readable
    icon_dict = {}
    icons = [icon for icon in df.icon.unique() if '_CIRCLES' in icon]
    icons = ['_'.join(icon.split('_')[-3:]).replace('.png', '') for icon in icons]
    for i, icon in enumerate(icons):
        side, id_1, id_2 = icon.split('_')
        text = df.loc[[re.search(icon + '.png', x) is not None for x in df.icon]].iloc[0]['name']
        icon_dict[i] = {'side':side, 'id_1':id_1, 'id_2':id_2, 'text':text}

    icon_df = pd.DataFrame(icon_dict).T
    icon_df.id_2 = icon_df.id_2.map(lambda x: '0' * (3 - len(x)) + x)
    icon_df['event_type'] = icon_df.id_2.map(event_labels.get)

    # Get the front line data and save separately
    front_line = df.loc[lambda x: [xx.endswith('Front line') for xx in x.name], :].copy()

    # Convert the front to a geometry
    front_geo = geopandas.GeoSeries(front_line.coords.apply(LineString), crs='EPSG:4326')
    front_geo.name = 'geometry'

    front_line = front_line.drop(columns=['coords', 'icon']).join(front_geo)
    front_line = geopandas.GeoDataFrame(front_line, crs='EPSG:4326')
    front_line['date'] = pd.to_datetime(front_line.name.map(extract_date), format='%d %H%M %b %Y')

    # The first 85 are icon labels (drop), and also drop front line polygons
    df = df.loc[85:].loc[lambda x: [not xx.endswith('Front line') for xx in x.name], :].copy()

    # Convert the date 
    df['date'] = pd.to_datetime(df.name.map(extract_date).iloc[:-2], format='%d %H%M %b %Y')

    # Convert the "tags", the type of media confirmed from twitter
    df = df.join(pd.DataFrame(df.name.map(get_tags).values.tolist(), index=df.index))

    # Strip date and tag info from the "name" column
    df.name = df.name.map(clean_name)

    # Save what's left as a comment from geoconfirmed (as opposed to the original twitter comment)
    df.rename(columns={'name':'geoconfirmed_disc'}, inplace=True)
    df['geoconfirmed_disc'] = df['geoconfirmed_disc'].replace('', np.nan)

    # Get out the icon data and merge in the labels
    df = df.join(df.icon.str.replace('.png', '', regex=False)
            .str.split('_').str[-3:]
            .loc[lambda x: x.str.len() == 3] #drops some "other" icons
            .map(lambda x: {'side': x[0], 'id_1':x[1], 'id_2':x[2]})
            .apply(pd.Series)
            .assign(id_2 = lambda x: (3 - x.id_2.map(len)).map(lambda y: '0' * y) + x.id_2)
            .assign(event_label = lambda x: x.id_2.map(event_labels.get)))


    # Extract URLs from the tweet text. These include the GeoConfirmed tweets announcing
    # the event, and the original source tweet.
    url_df = (df.twitter_source
              .apply(lambda x: get_urls(x, GEOTWEET_URL_PATTERN, 'geo_url'))
              .apply(pd.Series)
              .join(df.twitter_source
                    .apply(lambda x: erase_urls(x, GEOTWEET_URL_PATTERN))
                    .apply(lambda x: get_urls(x, SOURCE_URL_PATTERN, 'source_url'))
                    .apply(pd.Series)))
    df = df.join(url_df)

    # Clean the summary of the original tweet text 
    df = (df.assign(twitter_disc = lambda x: x.twitter_source
                                .str.replace('<br>', '')
                                .str.replace('Geo:', '')
                                .apply(lambda x: erase_urls(x, SOURCE_URL_PATTERN))
                                .apply(erase_tags)
                                .str.replace('<br>', '')
                                .str.strip())
          .replace({'':np.nan}))

    # Extract lat/lon data and convert to geopandas
    df = df.join((df.coords.str[0]
                  .apply(lambda x: pd.Series({'lat':x[0], 'lon':x[1]}))
                  .assign(geometry = lambda x: geopandas.points_from_xy(x=x['lat'], y=x['lon']))))
    df = geopandas.GeoDataFrame(df, crs='EPSG:4326')

    # Get oblast info by merging with a shapefile
    ukraine = geopandas.read_file('data/shapefiles/UKR/UKR_adm2.shp')
    data = []
    for idx, row in df.iterrows():
        point = row.geometry
        region_mask = ukraine.contains(point)
        in_ukraine = region_mask.any()
        if not in_ukraine:
            continue
        region = ukraine.loc[region_mask]    
        if region.shape[0] == 1:
            region_data = {'index':idx,
                           'country':region.ISO.values[0],
                           'name_1':region.NAME_1.values[0],
                           'name_2':region.NAME_2.values[0],
                           'varname':region.VARNAME_2.values[0]}
            data.append(region_data)
        elif region.shape[0] > 1:
            assert False

    region_df = pd.DataFrame(data).set_index('index')
    df = df.join(region_df, how='right').sort_index()

    # Put the columns in a nice order and drop: coords, icon, twitter_source, id_1, id_2 (these have been consumed)
    # to make other columns
    df = df[['date', 'side', 'event_label', 'country', 'name_1', 'name_2', 'varname', 'lat', 'lon', 'geoconfirmed_disc', 'twitter_disc', 'VID',
           'UAV', 'PIC', 'SAT', 'DRONE'] + [x for x in df if '_url_' in x] + ['geometry']]

    # Project to EPSG:6381, Ukraine TM zome 7, in meters Easting/Northing
    df = df.to_crs('EPSG:6381')
    ukraine = ukraine.to_crs('EPSG:6381')
    front_line = front_line.to_crs('EPSG:6381')
    
    return df, ukraine, front_line