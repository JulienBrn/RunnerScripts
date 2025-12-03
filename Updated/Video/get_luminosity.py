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
        - Goal: Get average luminosity of shapes in a video. The shapes are defined by the labelstudio annotations 
        - Output: an xarray of num_frames*num_shapes
    """
    video_path: Annotated[Path, Field(
        description="Path to the video", 
        default="/media/filer2/T4b/....mp4", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".mp4"]))
    )]
    max_n_frames: Annotated[int | None,  Field(
        default=None,
        description="max number of frames to process, usually used for testing",
    )]
    
    annotation_num: Annotated[int,  Field(
        description="Annoation number in label studio",
        default=1797, 
    )]

    output_path: Annotated[Path | None,  Field(
        description="Path to the result, may be empty if only image is demanded",
        default=["/media/filer2/T4b/...xr.zarr"], 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".xr.zarr"]))
    )]    
    
    fig_output_path: Annotated[Path | None,  Field(
        description="Path to the figure containing the first image with mask, may be None",
        default="/media/filer2/T4b/....html", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".html"]))
    )]
    
    allow_output_overwrite: bool = Field(default=False, description="If yes, erases the outputs if they exists before starting computation")
    _run_info: ClassVar = dict(cpu=0.1, gpu=0.0, memory=0.1)


args = Args()

from dafn.video_utilities import get_luminosity

with check_output_paths([args.fig_output_path, args.output_path], args.allow_output_overwrite) as [fig_output_path, output_path]:

    luminosities = get_luminosity(args.annotation_num, args.video_path, fig_output_path, args.max_n_frames)

    luminosities.rename(led_name="channel").to_zarr(output_path)