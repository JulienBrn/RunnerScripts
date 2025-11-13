from pathlib import Path
from pydantic import Field, BaseModel
from script2runner import CLI
from typing import List, Literal, Dict, ClassVar, Annotated
import re
from dafn.runner_helper import get_file_pattern_from_suffix_list, check_output_paths
from dafn.tool_converter import BinaryFamilyProcessInfo, EventFamilyProcessInfo, basic_poly_processing_family

class Args(CLI):
    """
        - Goal: Generates an event file from a poly task file and poly dat file
    """
    dat_path: Annotated[Path, Field(
        description="Path to the dat file",
        default="/media/filer2/T4b/Temporary/....dat", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".dat"]))
    )]
    task_path: Annotated[Path, Field(
        description="Path to the dat file",
        default="/media/filer2/T4b/Temporary/....xls", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".xls"]))
    )]
    dat_family_processing: Dict[int, Dict[str, BinaryFamilyProcessInfo | EventFamilyProcessInfo]] = basic_poly_processing_family

    output_path: Annotated[Path, Field(
        default="/media/filer2/T4b/Temporary/....xlsx", 
        description="Where you want your event file", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".xlsx"]))
    )]
    allow_output_overwrite: Literal["yes", "no"] = Field(default="no", description="Whether to overwrite and continue if output exists")
    _run_info: ClassVar = dict(cpu=0.1, memory=0.1, gpu=0.0, disk=0.6)
    
args = Args()
import pandas as pd, numpy as np
from dafn.tool_converter import polydat2df, convert2events
from dafn.runner_helper import finalize_events



with check_output_paths(args.output_path, args.allow_output_overwrite) as output_path:
    event_df = polydat2df(args.dat_path, args.task_path)
    final_df = convert2events(event_df, args.dat_family_processing)
    finalize_events(final_df, output_path)
    