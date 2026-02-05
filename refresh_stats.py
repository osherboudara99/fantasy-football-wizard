import nflreadpy as nfl 
import pandas as pd


def convert_polars_to_pandas(df):
    return df.to_pandas()

def null_removal(df, columns):
    return df.dropna(subset=columns)

def raw_regular_season_ingestion():
    return nfl.load_player_stats(seasons=True, summary_level='reg')


def seasonal_data_ingestion():

    seasonal = nfl.load_player_stats(seasons=True, summary_level='reg')
    seasonal = convert_polars_to_pandas(seasonal)
    seasonal = null_removal(df=seasonal, columns=['player_id', 'position', 'recent_team'])
    seasonal.to_csv('')
    



weekly = nfl.load_player_stats(summary_level="week").to_pandas()

