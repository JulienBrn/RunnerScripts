from pathlib import Path
from pydantic import Field, BaseModel
from script2runner import CLI
from typing import List, Literal, Dict, ClassVar, Annotated
import shutil, datetime as dt
import yaml
import re
import subprocess, sys

start_path_patterns = ["/media/filer2/T4b/", "/media/filer2/T4/", "/media/t4user/data1/", "/media/BigNAS/", "/home/t4user/"]
def get_file_pattern_from_suffix_list(start_path_patterns, suffixes):
    def mk_or_pattern(options):
        return '(('+ ")|(".join([re.escape(opt) for opt in options])+ '))'
    return '^'+mk_or_pattern(start_path_patterns)+r'[^\\]*'+mk_or_pattern(suffixes) + "$"
    
class Slicing(BaseModel):
    start: float | None = None
    end: float | None = None

class Args(CLI):
    """
        - Goal: Preprocesses a spikeGLX recording with [spikeinterface](https://spikeinterface.readthedocs.io/en/stable/). 
        - Technique: It applies phase shift, applies bandpass filter, removes bad channels, applies common reference. 
        - Note: This script may also slice the recording. This is mainly used for testing purposes to generate small data.
    """
    spike_glx_file: Annotated[Path, Field(
        description="Path to the spikeglx bin file containing the channel to preprocess", 
        examples=["/media/filer2/T4b/...imec0.ap.bin"], 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".bin"]))
    )]
    output_path: Annotated[Path,  Field(
        examples=["/media/t4user/data1/Data/SpikeSorting/...si.zarr"], 
        description="Location of the output",
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(["/media/t4user/data1/"], [".si.zarr"]))
    )]
    overwrite: Literal["yes", "no"] = Field(default="no", description="Whether to overwrite and continue if output exists")
    slice_recording: Slicing = Field(default=Slicing(), description="If you need to cut the recording given start and end times in seconds")
    _run_info: ClassVar = dict(conda_env="ssnew", cpu_usage=5.0, memory_usage=10.0)
    
a = Args()

import spikeinterface as si
import spikeinterface.extractors as sie
import spikeinterface.preprocessing as sip

if a.output_path.exists():
    if a.overwrite =="no":
        print(f"{a.output_path} already exists")
        exit(2)
    else:
        shutil.rmtree(a.output_path)
        a.output_path.parent.mkdir(exist_ok=True, parents=True)
        


rec : si.BaseRecording = sie.read_spikeglx(folder_path = a.spike_glx_file.parent, stream_name="".join(a.spike_glx_file.suffixes[:-1])[1:])

if rec.get_num_segments() != 1:
    raise Exception("Only single segment recordings supported")
if a.slice_recording.start or a.slice_recording.end:
    rec : si.BaseRecording = rec.frame_slice(
        a.slice_recording.start * rec.sampling_frequency if a.slice_recording.start else None, 
        a.slice_recording.end* rec.sampling_frequency if a.slice_recording.end else None
    )
    print("Slicing done")
print(rec)
sys.stdout.flush()
rec: si.BaseRecording = sip.phase_shift(rec)
rec = sip.bandpass_filter(rec, freq_max=6000, freq_min=300)
bad_channel_ids, _= sip.detect_bad_channels(rec, chunk_duration_s= 0.5, method= "coherence+psd", num_random_chunks= 100)
print(f"Detected the following bad channel: {bad_channel_ids}")
rec = rec.remove_channels(bad_channel_ids)
rec = sip.common_reference(rec, local_radius=[50, 100], reference= "local")
tmp_result_path = a.output_path.with_stem(".tmp_"+a.output_path.stem)
if tmp_result_path.exists():
    shutil.rmtree(tmp_result_path)
rec.save(format="zarr", folder=tmp_result_path, **{"n_jobs": 10,"chunk_duration": "1s","progress_bar": True}, channel_chunk_size=5)
try:
    shutil.move(tmp_result_path, a.output_path)
except Exception as e: #Stupid problem due to cifs mount
    print(f"Got exception {e} when trying to move result, checking that result is ok anyway")
    p = subprocess.run(["diff", "-r", tmp_result_path, a.output_path], check=True, stdout=subprocess.PIPE, text=True)
    if p.stdout.strip()=="":
        print("No differences found...")
        shutil.rmtree(tmp_result_path)
    else:
        print(f"Differences found {p.stdout}")
        exit(2)
