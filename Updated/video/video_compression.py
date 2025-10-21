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
        description="Path to the video", 
        default="/t4servershared/....mp4", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".mp4", ".avi"]))
    )]
    output: Path = Field(..., 
        description="Path of where to save the video.", 
        default="/t4servershared/....mp4", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".mp4"]))
    )
    crf: int = Field(13, description="Constant rate factor for compression. Higher values mean lower quality but smaller file size. Sane values are between 18 and 28", gt=0, lt=51)   
    overwrite: Literal["yes", "no", "ask_user"] = "no"
    slice: Slice = Slice()
    _run_info: ClassVar = dict(cpu=5.0, memory=3.0)

args = Args()

import subprocess
from dafn.video_utilities import compress_video

with check_output_paths(args.output, args.overwrite) as output_path:
    compress_video(args.video_path, output_path, args.crf, args.slice.start, args.slice.duration)
