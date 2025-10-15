from pathlib import Path
from pydantic import Field, BaseModel
from script2runner import CLI
from typing import List, Literal, Dict, ClassVar, Annotated
import re


start_path_patterns = ["/media/filer2/T4b/", "/media/filer2/T4/", "/media/t4user/data1/", "/media/BigNAS/", "/home/t4user/"]
def get_file_pattern_from_suffix_list(start_path_patterns, suffixes):
    def mk_or_pattern(options):
        return '(('+ ")|(".join([re.escape(opt) for opt in options])+ '))'
    return '^'+ mk_or_pattern(start_path_patterns)+r'[^\\]*'+ mk_or_pattern(suffixes) + "$"

class Args(CLI):
    """
        - Goal: Reads the eeg file. You may need to store in local storage and not on filer (problems with h5 on shared filesystems)
    """
    eeg_path: Annotated[Path, Field(
        description="Path to the file contining the eeg recording",
        examples=["/media/t4user/data1/Data/SpikeSorting/....bdf"], 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".edf", ".bdf"]))
    )]
    output_path: Annotated[Path, Field(
        examples=["/media/filer2/T4b/Temporary/....xr.h5"], 
        description="Where you want your output file", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".xr.h5"]))
    )]
    overwrite: Literal["yes", "no"] = Field(default="no", description="Whether to overwrite and continue if output exists")
    _run_info: ClassVar = dict(conda_env="dbscripts", uses_gpu=False)
    
a = Args()


if a.output_path.exists():
    if a.overwrite =="yes":
        a.output_path.unlink()
    else:
        raise Exception("Output path already exists")


import pandas as pd
import mne
import dask.array as da, xarray as xr
import dask, tqdm.auto as tqdm

eeg_data = mne.io.read_raw_bdf(a.eeg_path)
EEG_chans = [d["ch_name"] for d in eeg_data.info["chs"] if d["kind"] ==2]
times = eeg_data.times


read_progress = tqdm.tqdm(desc="Reading mne file")

def read_chunk(chan_index, t_index, chunk_size):
    ret, _ = eeg_data[EEG_chans[chan_index], t_index:t_index+chunk_size]
    read_progress.update()
    return ret

channel_chunks = []
n_chunks=0
for chan_index in range(0, len(EEG_chans), 1):
    time_chunks = []
    for t_index in range(0, times.size, 10**6):
        n_chunks+=1
        chunk_size = min(10**6, times.size - t_index)
        chunk = da.from_delayed(dask.delayed(read_chunk)(chan_index, t_index, chunk_size),
            shape=(1, ) + (chunk_size, ),
            dtype=float
        )
        time_chunks.append(chunk)
    channel_chunks.append(da.concatenate(time_chunks, axis=1))
darr = da.concatenate(channel_chunks, axis=0)
read_progress.total = n_chunks

ds = xr.Dataset()
ds["channel"] = xr.DataArray(EEG_chans, dims="channel")
ds["t"] = xr.DataArray(times, dims="t")
ds["data"] = xr.DataArray(darr, dims=["channel", "t"])

print(ds)
ds.to_netcdf(a.output_path, engine="h5netcdf")