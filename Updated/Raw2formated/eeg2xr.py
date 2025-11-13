from pathlib import Path
from pydantic import Field, BaseModel
from script2runner import CLI
from typing import List, Literal, Dict, ClassVar, Annotated
import re
from dafn.runner_helper import get_file_pattern_from_suffix_list, check_output_paths

class Args(CLI):
    """
        - Goal: Reads the eeg file. You may need to store in local storage and not on filer (problems with h5 on shared filesystems)
    """
    eeg_path: Annotated[Path, Field(
        default="/t4servershared/....bdf", 
        description="Path to the file contining the eeg recording", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".edf", ".bdf"]))
    )]
    output_path: Annotated[Path, Field(
        default="/t4servershared/....xr.zarr", 
        description="Where you want your output file", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".xr.zarr"]))
    )]
    allow_output_overwrite: bool = Field(default=False, description="If yes, erases the outputs if they exists before starting computation")
    _run_info: ClassVar = dict(conda_env="mne", cpu=0.1, memory=2, gpu=0.0, disk=3)
    
args = Args()

import mne
from dafn.tool_converter import eeg2xr

with check_output_paths(args.output_path, args.allow_output_overwrite) as output_path:
    eeg = mne.io.read_raw_bdf(args.eeg_path)
    ds = eeg2h5(eeg)
    ds.to_zarr(output_path)



