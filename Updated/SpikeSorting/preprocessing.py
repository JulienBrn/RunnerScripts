from pathlib import Path
from pydantic import Field, BaseModel
from script2runner import CLI
from typing import List, Literal, Dict, ClassVar, Annotated


from dafn.runner_helper import get_file_pattern_from_suffix_list, check_output_paths
    
class Slicing(BaseModel):
    start: float | None = None
    end: float | None = None

class Params(BaseModel):
    filter_low_freq: float = 300
    filter_high_freq: float = 6000

class Args(CLI):
    """
        - Goal: Preprocesses a spikeGLX recording with [spikeinterface](https://spikeinterface.readthedocs.io/en/stable/). 
        - Technique: It applies phase shift, applies bandpass filter, removes bad channels, applies common reference. 
        - Note: This script may also slice the recording. This is mainly used for testing purposes to generate small data.
    """
    spike_glx_file: Annotated[Path, Field(
        description="Path to the spikeglx bin file containing the channel to preprocess", 
        default="/media/filer2/T4b/...imec0.ap.bin", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".bin"]))
    )]
    params: Params = Field(default=Params(), description="Parameters of the preprocessing")
    slice_recording: Slicing = Field(default=Slicing(), description="If you need to cut the recording given start and end times in seconds")
    output_path: Annotated[Path,  Field(
        default="/media/t4user/data1/Data/SpikeSorting/...si.zarr", 
        description="Location of the output",
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".si.zarr"]))
    )]
    allow_output_overwrite: bool = Field(default=False, description="If yes, erases the outputs if they exists before starting computation")
    
    _run_info: ClassVar = dict(conda_env="ss", cpu=10.0, memory=5.0)
    
args = Args()

from dafn.spike_sorting import preprocess
import spikeinterface.extractors as sie

with check_output_paths(args.output_path, args.allow_output_overwrite) as output_path:
    rec = sie.read_spikeglx(folder_path = args.spike_glx_file.parent, stream_name="".join(args.spike_glx_file.suffixes[:-1])[1:])
    rec, bad_channel_ids = preprocess(rec, args.slice_recording.start, args.slice_recording.end, 
                                      args.params.filter_low_freq, args.params.filter_high_freq, 100)
    print(f"Detected the following bad channel: {bad_channel_ids}")
    rec.save(format="zarr", folder=output_path, **{"n_jobs": 10,"chunk_duration": "1s","progress_bar": True}, channel_chunk_size=5)

