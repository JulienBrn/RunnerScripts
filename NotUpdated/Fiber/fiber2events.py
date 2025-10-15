from pathlib import Path
from pydantic import Field, BaseModel
from script2runner import CLI
from typing import List, Literal, Dict, ClassVar, Annotated
import re


start_path_patterns = ["/media/filer2/T4b/", "/media/filer2/T4/", "/media/t4user/data1/", "/media/BigNAS/", "/home/t4user/"]
def get_file_pattern_from_suffix_list(start_path_patterns, suffixes):
    def mk_or_pattern(options):
        return '(('+ ")|(".join([re.escape(opt) for opt in options])+ '))'
    return '^'+ mk_or_pattern(start_path_patterns)+r'[^\\]*'+ mk_or_pattern(suffixes) + "$"

class Args(CLI):
    """
        - Goal: Reads the fiber event file and extracts events
    """
    fiber_path: Annotated[Path, Field(
        description="Path to the file contining the fiber events",
        examples=["/media/t4user/data1/...Events.csv"], 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".csv"]))
    )]
    output_path: Annotated[Path, Field(
        examples=["/media/filer2/T4b/Temporary/....xlsx"], 
        description="Where you want your output excel file", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".xlsx"]))
    )]
    overwrite: Literal["yes", "no"] = Field(default="no", description="Whether to overwrite and continue if output exists")
    _run_info: ClassVar = dict(conda_env="dbscripts", uses_gpu=False)
    
a = Args()

if a.output_path.exists():
    if a.overwrite =="yes":
        a.output_path.unlink()
    else:
        raise Exception("Output path already exists")
    
a.output_path.parent.mkdir(parents=True, exist_ok=True)

import pandas as pd, numpy as np

# Load CSV
df = pd.read_csv(a.fiber_path)
df["t"] = df["TimeStamp"]/1000
if not df["t"].is_monotonic_increasing:
    raise Exception("Data should already be sorted")
df = df.reset_index()

res = []
for n, g in df.groupby("Name"):
    starts_arr = g.loc[g["State"] == 0, "t"].to_numpy()
    ends_arr = g.loc[g["State"] == 1, "t"].to_numpy()
    start_indices = g.loc[g["State"] == 0, "index"].to_numpy()

    if starts_arr.size > 0 and ends_arr.size > 0:
        if starts_arr[0] > ends_arr[0]:
            starts_arr = np.insert(starts_arr, 0, np.nan)
            start_indices = np.insert(start_indices, 0, -1)
        if starts_arr[-1] > ends_arr[-1]:
            ends_arr = np.append(ends_arr, np.nan)

    if starts_arr.shape != ends_arr.shape:
        raise Exception("Not same number of rise and fall events")
    durations = ends_arr - starts_arr
    if (durations < 0 ).any():
        raise Exception("Problem aligning rise and falls events")
    result = pd.DataFrame()
    result["start"] = starts_arr
    result["duration"] = durations
    result["event_name"] = n
    result["index"] = start_indices
    res.append(result)

final_df = pd.concat(res).sort_values("index")[["event_name", "start", "duration"]].reset_index(drop=True)
final_df

print(final_df)
print("Counts are: ")
print(final_df.groupby("event_name").size())

final_df.to_excel(a.output_path, index=False)