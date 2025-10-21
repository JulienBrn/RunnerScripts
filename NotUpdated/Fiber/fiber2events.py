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
    _run_info: ClassVar = dict(uses_gpu=False)
    
a = Args()

from dafn.tool_converter import fiber2events
if a.output_path.exists():
    if a.overwrite =="yes":
        a.output_path.unlink()
    else:
        raise Exception("Output path already exists")
    
a.output_path.parent.mkdir(parents=True, exist_ok=True)

import pandas as pd, numpy as np
df = pd.read_csv(a.fiber_path)
final_df = fiber2events(df)

print(final_df)
print("Counts are: ")
print(final_df.groupby("event_name").size())

final_df.to_excel(a.output_path, index=False)