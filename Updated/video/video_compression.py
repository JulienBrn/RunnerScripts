import json, shutil
from pathlib import Path
import tqdm.auto as tqdm
from typing import List, Literal, Dict, ClassVar, Annotated
import re
from pydantic import Field, BaseModel
from script2runner import CLI
from dafn.runner_helper import get_file_pattern_from_suffix_list, check_output_paths


class Slice(BaseModel):
    start: float | None  = Field(None, description="Start time of the slice in seconds")
    duration: float | None = Field(None, description="End time of the slice in seconds")


class Args(CLI):
    video_path: Annotated[Path, Field(
        default="/t4servershared/....mp4", 
        description="Path to the video", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".mp4", ".avi"]))
    )]
    output: Annotated[Path, Field(
        default="/t4servershared/....mp4", 
        description="Path of where to save the video.", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".mp4"]))
    )]
    slice: Slice = Slice()
    crf: int = Field(13, description="Constant rate factor for compression. Higher values mean lower quality but smaller file size. Sane values are between 18 and 28", gt=0, lt=51)   
    allow_output_overwrite: bool = Field(default=False, description="If yes, erases the outputs if they exists before starting computation")
    _run_info: ClassVar = dict(cpu=5.0, memory=3.0)

args = Args()

from dafn.video_utilities import compress_video

with check_output_paths(args.output, args.allow_output_overwrite) as output_path:
    compress_video(args.video_path, output_path, args.crf, args.slice.start, args.slice.duration)
