# from pathlib import Path
# from pydantic import Field, BaseModel
# from script2runner import CLI
# from typing import List, Literal, Dict, ClassVar, Annotated
# import re


# start_path_patterns = ["/media/filer2/T4b/", "/media/filer2/T4/", "/media/t4user/data1/", "/media/BigNAS/", "/home/t4user/"]
# def get_file_pattern_from_suffix_list(start_path_patterns, suffixes):
#     def mk_or_pattern(options):
#         return '(('+ ")|(".join([re.escape(opt) for opt in options])+ '))'
#     return '^'+ mk_or_pattern(start_path_patterns)+r'[^\\]*'+ mk_or_pattern(suffixes) + "$"

# class Args(CLI):
#     """
#         - Goal: Converts a blackrock file into a electrophy h5 file and a event h5 file (splitting on channels)
#     """
#     nsx_path: Annotated[Path, Field(
#         description="Path to the file contining the recording",
#         examples=["/media/filer2/T4b/Temporary/....ns5"], 
#         json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".ns5"]))
#     )]
#     event_chanid_start:  Annotated[int | None, Field(
#         description="Start id of event channels",
#         examples=[97]
#     )]
#     event_chanid_end:  Annotated[int | None, Field(
#         description="End id of event channels",
#         examples=[None]
#     )]
#     output_event_path: Annotated[Path, Field(
#         examples=["/media/filer2/T4b/Temporary/....h5"], 
#         description="Where you want your event file", 
#         json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".xr.h5"]))
#     )]
#     output_electrophy_path: Annotated[Path, Field(
#         examples=["/media/filer2/T4b/Temporary/....h5"], 
#         description="Where you want your electrophy file", 
#         json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".xr.h5"]))
#     )]
#     overwrite: Literal["yes", "no"] = Field(default="no", description="Whether to overwrite and continue if output exists")
#     _run_info: ClassVar = dict(conda_env="monkey", uses_gpu=False)
    
# a = Args()