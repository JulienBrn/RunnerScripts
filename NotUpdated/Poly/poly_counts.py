from pathlib import Path
from pydantic import Field, BaseModel
from script2runner import CLI
from typing import List
import pandas as pd

class Counter(BaseModel):
    poly_line_num: int = Field(..., description="Task line number for which we count the number of times we went through", gt=0)
    name: str = Field(..., description="Name to give to that counter", examples=["my_counter"])
    
class Args(CLI):
    input_folder: Path = Field(...,description="Path to the folder containing the poly dat files to analyze", examples=["/media/filer2/T4b/..."])
    glob_pattern: str = Field("**/*.dat", description="pattern to use to look for dat files")
    counters: List[Counter] = Field(...,description="Your counters", examples=[[Counter(poly_line_num=2, name="my_counter")]]) 
    output_file: Path = Field("/media/filer2/T4b/Temporary/counts.tsv", description="Path to the result file")

params = Args()

files = params.input_folder.glob(params.glob_pattern)
all_dfs = []
file_start_row = pd.DataFrame([{'time (ms)':0, 'family':10, '_T':-1}])
for file in files:
  event_df = pd.read_csv(file, sep="\t", names=['time (ms)', 'family', 'nbre', '_P', '_V', '_L', '_R', '_T', '_W', '_X', '_Y', '_Z'], skiprows=13, dtype=int)
  all_dfs+=[file_start_row, event_df]
all_dfs = pd.concat(all_dfs, ignore_index=True)

line_change_df = all_dfs.loc[all_dfs["family"]==10, ['time (ms)', '_T']]
if (line_change_df["_T"] == line_change_df["_T"].shift(-1, fill_value=pd.NA)).any():
    raise Exception("Filtering issue")

res=[]
for counter in params.counters:
    res.append((counter.name, len(line_change_df.loc[line_change_df["_T"] == counter.poly_line_num].index)))
res = pd.DataFrame(res, columns=["counter_name", "count"])
print(res)
res.to_csv(params.output_file, sep="\t", index=False)