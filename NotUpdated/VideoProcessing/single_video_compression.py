import json, shutil
from pathlib import Path
import tqdm.auto as tqdm
from typing import List, Literal, Dict, ClassVar, Annotated
import re
from pydantic import Field, BaseModel
from script2runner import CLI


start_path_patterns = ["/media/filer2/T4b/", "/media/filer2/T4/", "/media/t4user/data1/", "/media/BigNAS/", "/home/t4user/"]
def get_file_pattern_from_suffix_list(start_path_patterns, suffixes):
    def mk_or_pattern(options):
        return '(('+ ")|(".join([re.escape(opt) for opt in options])+ '))'
    return '^'+mk_or_pattern(start_path_patterns)+r'[^\\]*'+mk_or_pattern(suffixes) + "$"


class Slice(BaseModel):
    start: float | None  = Field(None, description="Start time of the slice in seconds")
    duration: float | None = Field(None, description="End time of the slice in seconds")


class Args(CLI):
    video_path: Annotated[Path, Field(
        description="Path to the video", 
        examples=["/media/filer2/T4b/....mp4"], 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".mp4", ".avi"]))
    )]
    output: Path = Field(..., 
        description="Path of where to save the video.", 
        examples=["/media/filer2/T4b/Temporary/---.mp4"], 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".mp4"]))
    )
    crf: int = Field(13, description="Constant rate factor for compression. Higher values mean lower quality but smaller file size. Sane values are between 18 and 28", gt=0, lt=51)   
    overwrite: Literal["yes", "no", "ask_user"] = "no"
    slice: Slice = Slice()
    _run_info: ClassVar = dict(cpu_usage=5.0, memory_usage=3.0)

args = Args()

import subprocess

if args.output.exists():
    if args.overwrite != "yes":
        raise Exception(f"File {args.output} already exists")

tmp_file = args.output.with_suffix(".tmp.mp4")
if tmp_file.exists():
    tmp_file.unlink()

ffmpeg_args = ['ffmpeg', "-threads", "5", '-i', str(args.video_path), '-c:v', 'libx264', '-crf', str(args.crf), '-pix_fmt',"yuv420p"]
if args.slice.start:
     ffmpeg_args += ['-ss', str(args.slice.start)]
if args.slice.duration:
    ffmpeg_args += ['-t', str(args.slice.duration)]
ffmpeg_args += ['-y', str(tmp_file)]

try:
    subprocess.run(ffmpeg_args, stderr=subprocess.PIPE, stdout=subprocess.PIPE, stdin=subprocess.PIPE, check=True)
except subprocess.CalledProcessError as e:
    err : str= e.stderr.decode()
    s = err.lower().find("error")
    print(f"Error while running compression. Error is\n{err[s:]}")
    exit(2)

tmp_file.rename(args.output)




