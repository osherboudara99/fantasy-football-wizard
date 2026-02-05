import nflreadpy as nfl 
import pandas as pd


seasonal = nfl.load_player_stats(seasons=True, summary_level='reg').to_pandas()
weekly = nfl.load_player_stats(summary_level="week").to_pandas()

