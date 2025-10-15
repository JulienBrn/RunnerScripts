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
        - Goal: Reads the eeg file and extracts events
    """
    eeg_path: Annotated[Path, Field(
        description="Path to the file contining the eeg recording",
        examples=["/media/t4user/data1/Data/SpikeSorting/....bdf"], 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".edf", ".bdf"]))
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
import mne

eeg_data = mne.io.read_raw_bdf(a.eeg_path)
df = pd.DataFrame(mne.find_events(eeg_data,initial_event=True, output="step", consecutive=True), columns=["index", "from", "to"])
df["t"] = eeg_data.times[df["index"]]
df = df[["t", "from", "to"]]

df["from_next"] = df["from"].shift(-1, fill_value=0)
df["t_next"] = df["t"].shift(-1)
if (df["from_next"] != df["to"]).any():
    print(df)
    raise Exception("events are not isolated")
df["duration"] = df["t_next"] - df["t"]
df["event_name"] = df["to"]
df["start"] = df["t"]
df = df[["event_name", "start", "duration"]]

zero_df = df.iloc[1::2, :]
if (zero_df["event_name"] != 0).any():
    raise Exception("events do not always go back to 0")
final_df = df.iloc[::2, :]
if (final_df["event_name"] == 0).any():
    raise Exception("non-events found...")

print(final_df)
print("Counts are: ")
print(final_df.groupby("event_name").size())

final_df.to_excel(a.output_path, index=False)