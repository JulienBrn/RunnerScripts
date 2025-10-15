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
        examples=["/media/t4user/data1/....csv"], 
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

import pandas as pd

df = pd.read_csv(a.fiber_path)

input_names = df["Name"].unique()
durations_tab = {}

for input_ in input_names:
    # Filter and convert to float
    starts_ = df[(df["State"] == 0) & (df["Name"] == input_)].reset_index(drop=True)["TimeStamp"].astype(float)
    ends_ = df[(df["State"] == 1) & (df["Name"] == input_)].reset_index(drop=True)["TimeStamp"].astype(float)

    # Match lengths
    min_len = min(len(starts_), len(ends_))
    starts_ = starts_.iloc[:min_len]
    ends_ = ends_.iloc[:min_len]

    # Store durations
    durations_tab[input_] = ends_.values - starts_.values

starts = df[df["State"] == 0].reset_index(drop=True)
durations = []
event_name = {}
count_dic = {}
for input_ in starts["Name"]:
    count_dic[input_] = 0
for input_ in starts["Name"]:
    durations.append(durations_tab[input_][count_dic[input_]])
    count_dic[input_] += 1

# Build final table
task_table = pd.DataFrame({
    "start": starts["TimeStamp"].astype(float),
    "duration": durations,
    "event_name": starts["Name"]
})

task_table.to_excel(a.output_path, index=False)

print(task_table)