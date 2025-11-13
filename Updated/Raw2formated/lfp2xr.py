from pathlib import Path
from pydantic import Field, BaseModel
from script2runner import CLI
from typing import List, Literal, Dict, ClassVar, Annotated, List
import re, shutil
from dafn.runner_helper import get_file_pattern_from_suffix_list, check_output_paths

class Args(CLI):
    """
        # LFP JSON to Zarr Conversion Script — Summary
        This code reads a **Medtronic PERCEPT LFP JSON file**, extracts the **time-domain signals** from each channel, groups them by recording date, and saves each group as a **Zarr dataset** for analysis.
        ## Key Steps
        1. **Output Handling**  
        - Deletes the output folder if `overwrite="yes"`.
        2. **Signal Conversion**  
        - Converts each channel’s signal into an `xarray.DataArray` with a proper time axis and sampling frequency.
        3. **Grouping by Segment**  
        - Groups all channels from the same recording segment/date into one dataset.
        4. **Saving Output**  
        - Saves each segment as a separate `.xr.zarr` file using a formatted file name.
        ## Summary
        **Purpose:** Convert raw LFP JSON recordings into structured, time-indexed Zarr files suitable for analysis.
    """
    lfp_path: Annotated[Path, Field(
        default="/t4servershared/....json", 
        description="Path to the file contining the lfp recording",
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".json"]))
    )]
    output_path: Annotated[Path, Field(
        default="/t4servershared/..../myfolder", 
        description="Where you want your output files", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([""]))
    )]
    file_name_fstring: str = Field("{segnum}--{segdate}.xr.zarr", pattern=r".*\.xr\.zarr$")
    allow_output_overwrite: bool = Field(default=False, description="If yes, erases the outputs if they exists before starting computation")
    _run_info: ClassVar = dict(cpu=0.1, memory=0.1, gpu=0.)
    
args = Args()

from dafn.tool_converter import lfp2xr
import json

with check_output_paths(args.output_path, args.allow_output_overwrite) as output_path:
    lfp = json.load(args.lfp_path.open("r"))
    all_ds = lfp2h5(lfp)
    output_path.mkdir()
    for ds in all_ds:
        print(ds)
        ds.to_zarr(output_path/args.file_name_fstring.format_map(ds.attrs))

