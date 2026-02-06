import nflreadpy as nfl 
from pathlib import Path
from typing import Literal
import polars as pl
import pandas as pd


class Stats:
    def __init__(self, seasons: Literal['all', 'current'] | list[int], 
                 aggregation: Literal['week', 'reg', 'post', 'reg+post']): 
        if isinstance(seasons, str):
            if seasons == 'all':
                seasons_source_arg = True
            elif seasons == 'current':
                seasons_source_arg = None 
            else:
                raise ValueError(f"Invalid seasons string: {seasons}. Must be 'all', 'current' or list[int].")
        elif isinstance(seasons, list):
            seasons_source_arg = seasons
        else:
            raise TypeError(f"seasons must be str or list[int]")

        self.data = nfl.load_player_stats(seasons=seasons_source_arg, summary_level=aggregation)
    
    def null_removal(self, columns:list[str]) -> pd.DataFrame | pl.DataFrame:
        self.data = self.data.dropna(subset=columns)
        return self.data
    
    def null_summary(self):
        if isinstance(self.data, pd.DataFrame):
            return self.data.isnull().sum()
        elif isinstance(self.data, pl.DataFrame):
            return self.data.null_count()
        else:
            raise TypeError("Unsupported DataFrame type.")
        
    def save_parquet(self, file_name: str, stage: Literal["raw", "processed"] = "raw", base_dir: Path | str = "data/stats"):
        base_dir = Path(base_dir)

        stage_dirs = {
            "raw" : base_dir / "raw",
            "staged" : base_dir / "staged",
            "processed" : base_dir / "processed"
        }

        output_dir = stage_dirs[stage]
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / file_name

        if isinstance(self.data, pl.DataFrame):
            self.data.write_parquet(output_path)
        elif isinstance(self.data, pd.DataFrame):
            self.data.to_parquet(output_path)
        else:
            raise TypeError("self.data must be a Pandas or Polars DataFrame.")

        return output_path  


def seasonal(remove_nulls:bool, save_file:bool) -> None | str:
    seasonal = Stats(seasons='all', aggregation='reg')
    stage = 'raw'
    if remove_nulls:
        seasonal.null_removal(columns=['player_id', 'position', 'recent_team']) 
        stage = 'processed'
    if save_file:
        return seasonal.save_parquet(file_name='seasonal_data.parquet', stage=stage)


if __name__ == '__main__':
    seasonal

