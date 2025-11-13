from pathlib import Path
from pydantic import Field, BaseModel
from script2runner import CLI
from typing import List, Literal, Dict, ClassVar, Annotated
import re
from dafn.runner_helper import get_file_pattern_from_suffix_list, check_output_paths

class Args(CLI):
    """
        - Goal: Reads the eeg file and extracts events
    """
    eeg_path: Annotated[Path, Field(
        examples=["/media/t4user/data1/Data/SpikeSorting/....bdf"],
        description="Path to the file contining the eeg recording", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".edf", ".bdf"]))
    )]
    output_path: Annotated[Path, Field( 
        description="Where you want your output excel file",
        examples=["/media/filer2/T4b/Temporary/....xlsx"],
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".xlsx"]))
    )]
    allow_output_overwrite: Literal["yes", "no"] = Field(default="no", description="Whether to overwrite and continue if output exists")
    _run_info: ClassVar = dict(conda_env="mne", cpu=0.1, memory=2, gpu=0.0, disk=3)
    
args = Args()

import pandas as pd
import mne
from dafn.tool_converter import eeg2events

with check_output_paths(args.output_path, args.allow_output_overwrite) as output_path:
    eeg_data = mne.io.read_raw_bdf(args.eeg_path)
    df = pd.DataFrame(mne.find_events(eeg_data,initial_event=True, output="step", consecutive=True), columns=["index", "from", "to"])
    df["t"] = eeg_data.times[df["index"]]

    final_df = eeg2events(df)

    print(final_df)
    print("Counts are: ")
    print(final_df.groupby("event_name").size())

    final_df.to_excel(output_path, index=False)