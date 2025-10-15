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
        - Goal: Converts a blackrock file into a electrophy h5 file and a event h5 file (splitting on channels)
    """
    nsx_path: Annotated[Path, Field(
        description="Path to the file contining the recording",
        examples=["/media/filer2/T4b/Temporary/....ns5"], 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".ns5"]))
    )]
    event_chanid_start:  Annotated[int | None, Field(
        description="Start id of event channels",
        examples=[97]
    )]
    event_chanid_end:  Annotated[int | None, Field(
        description="End id of event channels",
        examples=[None]
    )]
    output_event_path: Annotated[Path, Field(
        examples=["/media/filer2/T4b/Temporary/....h5"], 
        description="Where you want your event file", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".xr.h5"]))
    )]
    output_electrophy_path: Annotated[Path, Field(
        examples=["/media/filer2/T4b/Temporary/....h5"], 
        description="Where you want your electrophy file", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".xr.h5"]))
    )]
    overwrite: Literal["yes", "no"] = Field(default="no", description="Whether to overwrite and continue if output exists")
    _run_info: ClassVar = dict(conda_env="monkey", uses_gpu=False)
    
a = Args()

for output_path in [a.output_electrophy_path, a.output_event_path]:
    if output_path.exists():
        if a.overwrite =="yes":
            output_path.unlink()
        else:
            raise Exception(f"Output path {output_path} already exists")

from brpylib import NsxFile, NevFile
import pandas as pd, xarray as xr, numpy as np
import dask.array as da
import dask
import tqdm.auto as tqdm

nsx_file = NsxFile(str(a.nsx_path))
cont_data = nsx_file.getdata("all", 0, "all", 1, full_timestamps=True)
ext_headers = pd.DataFrame(nsx_file.extended_headers).to_xarray().set_coords("ElectrodeID").rename(ElectrodeID="elec_id").drop_vars("index").rename(index="channel")

all_data = cont_data["data"][0]

read_progress = tqdm.tqdm(desc="Reading blackrock file")

def read_chunk(chan_index, t_index, chunk_size):
    ret = np.asarray(all_data[chan_index:chan_index+1, t_index:t_index+chunk_size])
    read_progress.update()
    return ret

channel_chunks = []
n_chunks=0
for chan_index in range(0, all_data.shape[0], 1):
    time_chunks = []
    for t_index in range(0, all_data.shape[1], 10**6):
        n_chunks+=1
        chunk_size = min(10**6, all_data.shape[1] - t_index)
        chunk = da.from_delayed(dask.delayed(read_chunk)(chan_index, t_index, chunk_size),
            shape=(1, ) + (chunk_size, ),
            dtype=all_data.dtype
        )
        time_chunks.append(chunk)
    channel_chunks.append(da.concatenate(time_chunks, axis=1))
darr = da.concatenate(channel_chunks, axis=0)
read_progress.total = n_chunks
d = xr.Dataset()
arr = xr.DataArray(darr, dims=["channel", "t"])
arr["t"] =  np.arange(all_data.shape[1]) / nsx_file.basic_header['SampleResolution'] 
arr["t"].attrs["fs"] = nsx_file.basic_header['SampleResolution'] 
d["data"] = arr
d["elec_id"] = xr.DataArray(cont_data["elec_ids"], dims="channel")
d = d.set_coords(["elec_id"])
d = xr.merge([d, ext_headers])
d.attrs["comment"] = nsx_file.basic_header["Comment"]
d.attrs["recording_date"] = str(nsx_file.basic_header["TimeOrigin"])
d["channel"] = d["elec_id"]
d = d.drop_vars("elec_id")

print(d)
events = d.sel(channel=slice(a.event_chanid_start, a.event_chanid_end))
electrophy = d.sel(channel=~d["channel"].isin(events["channel"]))
print(events)
print(electrophy)

events.to_netcdf(a.output_event_path, engine="h5netcdf")
electrophy.to_netcdf(a.output_electrophy_path, engine="h5netcdf")