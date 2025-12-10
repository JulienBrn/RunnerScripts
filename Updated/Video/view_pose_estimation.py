from pathlib import Path
from typing import List, Literal, Dict, ClassVar, Annotated
from pydantic import Field, BaseModel
from script2runner import CLI
from dafn.runner_helper import get_file_pattern_from_suffix_list, check_output_paths

class Args(CLI):
    """
    """
    pose_estimation_path: Annotated[Path, Field(
        default="path.xr.zarr", 
        description="Path to predictions"
    )]
    video_path: Annotated[Path, Field(
        description="Path to the video", 
        default="/media/filer2/T4b/....mp4", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".mp4"]))
    )]
    skeleton_path: Annotated[Path | None, Field(
        description="Path to the skeleton used", 
        default=None, 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".yaml"]))
    )]
    output_path: Annotated[Path, Field(
        description="The path of the result video",
        default="/media/filer2/T4b/UserFolders/Name/result-predict.mp4",
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".mp4"]))
    )]
    
    allow_output_overwrite: bool = Field(default=False, description="If yes, erases the outputs if they exists before starting computation")
    _run_info: ClassVar = dict(conda_env="dlc3", gpu=1)

args = Args()
import xarray as xr, yaml
from dafn.video_utilities import annotate_video

with check_output_paths(args.output_path, args.allow_output_overwrite) as output_path:
    pose_estimation = xr.load(args.pose_estimation_path).compute()
    if args.skeleton_path:
        with args.skeleton_path.open("r") as f:
            skeleton = yaml.safe_load(f)
    else:
        skeleton = None
    annotate_video(args.video_path, output_path, pose_estimation, skeleton)