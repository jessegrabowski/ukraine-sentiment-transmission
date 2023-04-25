from telethon.sync import TelegramClient
import cloudpickle as pickle
import yaml
import os
import numpy as np
import pandas as pd


OUT_DIR = 'data/telegram'
DICT_COLS = ['peer_id', 'action', 'from_id', 'reply_to', 'replies']


# Fill in a path to a .yaml file containing two items:
# api_key and api_hash (used by telethon to log into your account)
CREDENTIAL_PATH = None

# Fill in a dictionary with name:url key-value pairs
# for the channels you want to scrape
CHANNEL_DICT = None


def data_to_df(data, cols_to_process):
    df = pd.DataFrame(data)

    exploded_df = df.copy()
    for col in cols_to_process:
        col_data = df[col].copy()
        if col_data.isna().all():
            continue

        col_data[col_data.str.len() == 0] = np.nan
        col_keys = col_data.dropna().sample(1).values[0].keys()
        DEFAULT_DICT = {x: np.nan for x in col_keys}
        col_data = col_data.apply(lambda x: DEFAULT_DICT if pd.isna(x) else x)
        col_data = col_data.apply(pd.Series).drop(columns=['_'])
        new_cols = set(col_data.columns) - set(exploded_df.columns)
        new_cols = list(new_cols)
        exploded_df = exploded_df.drop(columns=[col]).join(col_data[new_cols])
    return exploded_df


async def get_data_for_channel(region, channel_name,
                               api_id, api_hash,
                               out_dir,
                               force_update=False,
                               pickle_draw_data=True,
                               save_dataframe=True,
                               convert_raw_to_df=True,
                               return_df=False):

    if channel_name is None:
        print(f'No channel data associated with {region}, skipping...')
        return

    if not force_update:
        completed_regions = [x.replace('.csv', '') for
                             x in os.listdir(out_dir)]
        if region in completed_regions:
            print(f'{region} data already downloaded, skipping...')
            return

    print(f'Getting data for {region} from {channel_name}')
    data = []
    i = 0
    async with TelegramClient('name', api_id, api_hash) as client:
        try:
            channel = await client.get_entity(channel_name)
            async for message in client.iter_messages(channel,
                                                      limit=None,
                                                      reverse=True):
                data.append(message.to_dict())
                i += 1
                if (i % 1000) == 0:
                    print(f'\t{i} messages accessed', end='\r')

        except ValueError as e:
            print(f'While trying to access {channel_name} encountered the '
                  f'following error:\n'
                  f'{e}\n\n'
                  'Continuing...')
    print('')

    if len(data) == 0:
        print('No data found? Skipping...')
        return

    if pickle_draw_data:
        with open(f'data/telegram/raw/{region}.p', 'wb') as file:
            pickle.dump(data, file)

    if convert_raw_to_df:
        df = data_to_df(data, DICT_COLS)

        if save_dataframe:
            df.to_csv(f'data/telegram/{region}.csv', index=False)
        if return_df:
            return df


with open(CREDENTIAL_PATH, 'r') as file:
    creds = yaml.safe_load(file)
api_id, api_hash = creds['api_id'], creds['api_hash']


# There's also entities, but it's a list of dictionaries; 
# don't want to deal with it.
async def run():
    for region, channel_name in CHANNEL_DICT.items():
        await get_data_for_channel(region, channel_name, api_id, api_hash,
                                   out_dir=OUT_DIR,
                                   force_update=False,
                                   pickle_draw_data=True,
                                   save_dataframe=True,
                                   convert_raw_to_df=True,
                                   return_df=False)

if __name__ == '__main__':
    await run()
