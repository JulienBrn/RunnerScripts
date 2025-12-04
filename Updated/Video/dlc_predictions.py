import json, shutil
from pathlib import Path
import tqdm.auto as tqdm
from typing import List, Literal, Dict, ClassVar, Annotated
import re
from pydantic import Field, BaseModel
from script2runner import CLI
from dafn.runner_helper import get_file_pattern_from_suffix_list, check_output_paths

class Args(CLI):
    """
    """
    model_path: Annotated[Path, Field(
        default="/media/filer2/T4b/UserFolders/Name/DLC-user-2025-03-13", 
        description="Path to the dlc model folder (folder containing config.yaml)"
    )]
    video_path: Annotated[Path, Field(
        description="Path to the video", 
        default="/media/filer2/T4b/....mp4", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".mp4"]))
    )]
    output_xarray_path: Annotated[Path, Field(
        description="The path where the results will be saved for later computation",
        default="/media/filer2/T4b/UserFolders/Name/result-predict.xr.zarr",
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".xr.zarr"]))
    )]
    output_excel_path: Annotated[Path, Field(
        description="The path where the results will be saved for human visualization",
        default="/media/filer2/T4b/UserFolders/Name/result-predict.xlsx",
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".xlsx"]))
    )]
    allow_output_overwrite: bool = Field(default=False, description="If yes, erases the outputs if they exists before starting computation")
    _run_info: ClassVar = dict(conda_env="dlc3", gpu=1)

args = Args()

import deeplabcut
import shutil
from dafn.video_utilities import dlc_predict

with check_output_paths([args.output_xarray_path, args.output_excel_path], args.allow_output_overwrite) as [output_xarray_path, output_excel_path]:
    result = dlc_predict(args.model_path, args.video_path)
    result.to_zarr(output_xarray_path)
    result.to_dataframe(name="value").reset_index().to_excel(output_excel_path)
