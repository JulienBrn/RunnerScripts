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

class BinaryFamilyProcessInfo(BaseModel):
    kind: Literal["binary"] = "binary"
    reverse: bool = False
    nbre_filter: List[int] = []
    def handle_group(self, group, state_col, name):
        print(f"called {self} {name}")
        group = group.loc[group[state_col] != group[state_col].shift(1)].copy()
        group[state_col] = group[state_col].astype(bool)
        if self.reverse:
            group[state_col] = ~group[state_col]
        if group[state_col].iat[0] == 0:
            group=group.iloc[1:, :]
        if len(group.index) == 0:
            return pd.DataFrame([], columns=["start", "duration", "event_name", "start_node", "end_node"])
        if (group[state_col].iloc[::2] != 1).any():
            print(group)
            raise Exception("Problem")
        if (group[state_col].iloc[1::2] != 0).any():
            print(group)
            raise Exception("Problem")
        rises = group["t"].iloc[::2]
        falls = group["t"].iloc[1::2].tolist()
        start_node = group["curr_node"].iloc[::2]
        end_node = group["curr_node"].iloc[1::2].tolist()
        if group[state_col].iat[-1] != 0:
            falls+=[np.nan]
            end_node+=[None]
        return pd.DataFrame().assign(start=rises, duration=falls-rises, event_name=name, start_node=start_node, end_node=end_node)
    
class EventFamilyProcessInfo(BaseModel):
    kind: Literal["event"] = "event"
    nbre_filter: List[int] = []
    def handle_group(self, group, state_col, name):
        group = group.loc[group[state_col].astype(bool)].copy()
        rises = group["t"]
        start_node = group["curr_node"]
        return pd.DataFrame().assign(start=rises, duration=np.nan, event_name=name, start_node=start_node, end_node=np.nan)




basic_poly_processing_family = {
    1: {"_P": BinaryFamilyProcessInfo()},
    2: {"_V": BinaryFamilyProcessInfo()},
    5: {"_P": EventFamilyProcessInfo()},
    6: {"_P": BinaryFamilyProcessInfo(reverse=True), "_V": BinaryFamilyProcessInfo(reverse=True, nbre_filter=[20])},
    13: {"_P": BinaryFamilyProcessInfo()},
    15: {"_P": BinaryFamilyProcessInfo()},
}
    

class Args(CLI):
    """
        - Goal: Generates an event file from a poly task file and poly dat file
    """
    dat_path: Annotated[Path, Field(
        description="Path to the dat file",
        examples=["/media/filer2/T4b/Temporary/....dat"], 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".dat"]))
    )]
    task_path: Annotated[Path, Field(
        description="Path to the dat file",
        examples=["/media/filer2/T4b/Temporary/....xls"], 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".xls"]))
    )]
    dat_family_processing: Dict[int, Dict[str, BinaryFamilyProcessInfo | EventFamilyProcessInfo]] = basic_poly_processing_family

    output_path: Annotated[Path, Field(
        examples=["/media/filer2/T4b/Temporary/....xlsx"], 
        description="Where you want your event file", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list(start_path_patterns, [".xlsx"]))
    )]
    overwrite: Literal["yes", "no"] = Field(default="no", description="Whether to overwrite and continue if output exists")
    _run_info: ClassVar = dict(conda_env="dbscripts", uses_gpu=False)
    
a = Args()
import pandas as pd, numpy as np

for output_path in [a.output_path]:
    if output_path.exists():
        if a.overwrite =="yes":
            output_path.unlink()
        else:
            raise Exception(f"Output path {output_path} already exists")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
dat_df = pd.read_csv(a.dat_path, sep="\t", names=['time (ms)', 'family', 'nbre', '_P', '_V', '_L', '_R', '_T', '_W', '_X', '_Y', '_Z'], skiprows=13, dtype=int)
dat_df["t"] = dat_df.pop('time (ms)')/1000
dat_df["curr_node"] =  dat_df["_T"].where(dat_df["family"]==10).ffill()

with Path(a.task_path).open("r") as f:
  i=0
  while(f):
      l = f.readline().split("\t")
      if len([x for x in l if "NEXT" in x]) >1:
          break
      i+=1
task_df = pd.read_csv(a.task_path, sep="\t", skiprows=i)

names = []
pat = r'(?P<name>\D+\d*)' + re.escape(r'(') + r'(?P<family>\d+),(?P<nbre>\d+)'+re.escape(')')
for c in task_df.columns:
    m= re.match(pat, c)
    if m:
        names.append(m.groupdict())
names = pd.DataFrame(names)
for col in ["family", "nbre"]:
    names[col] = names[col].replace('', None).astype(pd.Int64Dtype())

event_df = pd.merge(dat_df, names, on=["family", "nbre"], how="inner")





results = []
for (n, f, nb), group in event_df.groupby(["name", "family", "nbre"]):
    if not f in a.dat_family_processing:
        raise Exception(f"Unhandled family {f}")
    state_col = {k: v for k, v in a.dat_family_processing[f].items() if not nb in v.nbre_filter}
    if len(state_col) > 1:
        for s, g in state_col.items():
            results.append(g.handle_group(group, s, n+s))
    elif len(state_col) == 1:
        for s, g in state_col.items():
            results.append(g.handle_group(group, s, n))

results = pd.concat(results, ignore_index=True)

print(results.groupby("event_name").size())
results.sort_values("start").to_excel(a.output_path, index=False)