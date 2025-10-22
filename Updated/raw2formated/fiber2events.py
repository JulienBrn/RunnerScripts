from pathlib import Path
from pydantic import Field, BaseModel
from script2runner import CLI
from typing import List, Literal, Dict, ClassVar, Annotated
import re
from dafn.runner_helper import get_file_pattern_from_suffix_list, check_output_paths

class Args(CLI):
    """
        - Goal: Reads the fiber event file and extracts events
    """
    fiber_path: Annotated[Path, Field(
        default="/t4servershared/...Events.csv",
        description="Path to the file containing the fiber events",
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".csv"]))
    )]
    output_path: Annotated[Path, Field(
        default="/t4servershared/....xlsx", 
        description="Where you want your output excel file", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".xlsx"]))
    )]
    allow_output_overwrite: bool = Field(default=False, description="If yes, erases the outputs if they exists before starting computation")
    _run_info: ClassVar = dict(cpu=0.1, gpu=0.0, memory=0.1)
    
args = Args()

from dafn.tool_converter import fiber2events
import pandas as pd, numpy as np

with check_output_paths(args.output_path, args.allow_output_overwrite) as output_path:
    df = pd.read_csv(args.fiber_path)
    final_df = fiber2events(df)

    print(final_df)
    print("Counts are: ")
    print(final_df.groupby("event_name").size())

    final_df.to_excel(output_path, index=False)