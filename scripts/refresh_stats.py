import nflreadpy as nfl 
from pathlib import Path
from typing import Literal
import polars as pl
import pandas as pd


class Stats:
    def __init__(self, seasons: Literal['all', 'current'] | list[int], 
                 aggregation: Literal['week', 'reg', 'post', 'reg+post']): 
        self.output_paths: dict[str, Path] = {}
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
        self.stage = 'raw'
    
    def to_polars(self) -> "Stats":
        if isinstance(self.data, pd.DataFrame):
            self.data = pl.from_pandas(self.data)
        return self

    def to_pandas(self) -> "Stats":
        if isinstance(self.data, pl.DataFrame):
            self.data = self.data.to_pandas()
        return self

    
    def null_removal(self, columns:list[str]) -> "Stats":
        if isinstance(self.data, pd.DataFrame):
            self.data = self.data.dropna(subset=columns)
        elif isinstance(self.data, pl.DataFrame):
            self.data = self.data.drop_nulls(subset=columns)
        else:
            raise TypeError("Unsupported DataFrame type.")

        self.stage = 'staged'
        return self
    
    def null_summary(self) -> pd.Series | pl.DataFrame:
        if isinstance(self.data, pd.DataFrame):
            return self.data.isnull().sum()
        elif isinstance(self.data, pl.DataFrame):
            return self.data.null_count()
        else:
            raise TypeError("Unsupported DataFrame type.")
        
    def save_parquet(self, file_name: str, base_dir: Path | str = "data/stats") -> "Stats":
        base_dir = Path(base_dir)

        stage_dirs = {
            "raw" : base_dir / "raw",
            "staged" : base_dir / "staged",
            "processed" : base_dir / "processed"
        }

        output_dir = stage_dirs[self.stage]
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / file_name

        if isinstance(self.data, pl.DataFrame):
            self.data.write_parquet(output_path)
        elif isinstance(self.data, pd.DataFrame):
            self.data.to_parquet(output_path)
        else:
            raise TypeError("self.data must be a Pandas or Polars DataFrame.")
        
        
        self.output_paths[self.stage] = output_path
        print(f"Saved: {output_path}")
        return self

def weekly_stats() -> Stats:
    return (
        Stats(seasons="current", aggregation="week")
        # .to_pandas()
        .save_parquet("weekly_raw.parquet")
        .null_removal(["player_id", "position"])
        .save_parquet("weekly_staged.parquet")
        # .process()
        # .save_parquet("weekly_processed.parquet")
    )

def seasonal_stats() -> Stats:
    return (
        Stats(seasons="all", aggregation="reg")
        # .to_pandas()
        .save_parquet("seasonal_raw.parquet")
        .null_removal(["player_id", "position", "recent_team"])
        .save_parquet("seasonal_staged.parquet")
        # .process()
        # .save_parquet("seasonal_processed.parquet")
    )

if __name__ == '__main__':
    weekly = weekly_stats()
    seasonal = seasonal_stats()

    

