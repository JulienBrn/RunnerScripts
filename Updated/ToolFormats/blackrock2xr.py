from pathlib import Path
from pydantic import Field, BaseModel
from script2runner import CLI
from typing import List, Literal, Dict, ClassVar, Annotated
import re
from dafn.runner_helper import get_file_pattern_from_suffix_list, check_output_paths


class Args(CLI):
    """
        - Goal: Converts a blackrock file into a electrophy h5 file and a event h5 file (splitting on channels)
    """
    nsx_path: Annotated[Path, Field(
        default="/media/filer2/T4b/Temporary/....ns5", 
        description="Path to the file containing the recording",
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".ns5"]))
    )]
    event_chanid_start:  Annotated[int | None, Field(
        default=97,
        description="Start id of event channels"
    )]
    event_chanid_end:  Annotated[int | None, Field(
        default=None,
        description="End id of event channels"
    )]
    output_event_path: Annotated[Path, Field(
        default="/media/filer2/T4b/Temporary/....xr.zarr", 
        description="Where you want your event file", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".xr.zarr"]))
    )]
    output_electrophy_path: Annotated[Path, Field(
        default="/media/filer2/T4b/Temporary/....xr.zarr", 
        description="Where you want your electrophy file", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".xr.zarr"]))
    )]
    allow_output_overwrite: Literal["yes", "no"] = Field(default="no", description="Whether to overwrite and continue if output exists")
    _run_info: ClassVar = dict(cpu=0.1, memory=1, gpu=0.0, disk=3)
    
args = Args()

from brpylib import NsxFile
from dafn.tool_converter import blackrock2xr
import numpy as np
from dask.diagnostics import ProgressBar
import dask

with check_output_paths([args.output_electrophy_path, args.output_event_path], args.allow_output_overwrite) as [output_electrophy_path, output_event_path]:
    nsx_file = NsxFile(str(args.nsx_path))
    d = blackrock2xr(nsx_file)
    print(d)
    events = d.sel(channel=slice(args.event_chanid_start, args.event_chanid_end))
    electrophy = d.sel(channel=~d["channel"].isin(events["channel"]))
    
    with ProgressBar():
        ev = events.to_zarr(output_event_path, consolidated=False, zarr_format=2)
        elec = electrophy.to_zarr(output_electrophy_path, consolidated=False, zarr_format=2)
    