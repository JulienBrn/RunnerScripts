from pathlib import Path
from pydantic import Field, BaseModel
from script2runner import CLI
from typing import List, Literal, Dict, ClassVar, Annotated
import re
import abc, json
import xarray as xr, numpy as np, pandas as pd
from dafn.runner_helper import get_file_pattern_from_suffix_list, check_output_paths


class Args(CLI):
    """
## Inputs
- **Excel files**:
  - `reference_event_path` → the reference event log
  - `relative_event_path` → the event log to align with the reference
- **Event mapping** → maps event names between files (e.g., `LED1` → `LED_1`)
- **Paths** for:
  - Output JSON file
  - Output HTML figures
  - Optional debug folder

## Workflow
1. **Load & preprocess events**
   - Reads Excel files
   - Aligns event names using the mapping
   - Ensures uniqueness and chronological ordering
   - Converts into an `xarray` dataset

2. **Initial synchronization**
   - Estimates approximate alignment parameters:
     - **slope** (time scaling)
     - **shift** (time offset)
     - **tolerance** (matching window)

3. **Event matching**
   - Matches events between reference and relative logs within tolerance

4. **Refined synchronization**
   - Applies least-squares fitting for optimal slope and shift

5. **Compute statistics**
   - Counts matched, missed, and partially matched events

6. **Generate visualization**
   - Creates interactive **Plotly** figures:
     - Matching overview
     - Time alignment trend
     - Event stats
   - Saves as HTML

7. **Save outputs**
   - **JSON** → contains final sync parameters (`shift`, `slope`)
   - **HTML** → interactive visualization report

## Outputs
- `output_path` → JSON file with sync results
- `figure_path` → HTML file with plots

---

👉 **In short:**  
This script synchronizes two sets of event logs, aligns their timelines, computes matching statistics, and produces both a JSON summary and an interactive HTML report.

    """
    reference_event_path: Annotated[Path, Field(
        default="/media/filer2/T4b/Temporary/....xlsx", 
        description="Where you have your events", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".xlsx"]))
    )]
    relative_event_path: Annotated[Path, Field(
        default="/media/filer2/T4b/Temporary/....xlsx", 
        description="Where you have your events", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".xlsx"]))
    )]
    event_mapping: Dict[str | int, str | int] = {"LED1": "LED_1", "LED2":"LED_2", "LED3":"LED_3"}
    tolerance: float = 0.020
    output_path: Annotated[Path, Field(
        default="/media/filer2/T4b/Temporary/....json", 
        description="Where you want your output json file", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".json"]))
    )]
    figure_path: Annotated[Path, Field(
        default="/media/filer2/T4b/Temporary/....html", 
        description="Where you want your output figures", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([".html"]))
    )]
    debug_folder: Annotated[Path, Field(
        default="/media/filer2/T4b/Temporary/....", 
        description="Where you want your debug figures", 
        json_schema_extra=dict(pattern=get_file_pattern_from_suffix_list([]))
    )] | None = None
    
    allow_output_overwrite: bool = Field(default=False, description="If yes, erases the outputs if they exists before starting computation")
    _run_info: ClassVar = dict(cpu=0.1, gpu=0.0, memory=0.1)


args = Args()

from dafn.sync import sync

with check_output_paths([args.figure_path, args.output_path], args.allow_output_overwrite) as [figure_path, output_path]:
    ref_df = pd.read_excel(args.reference_event_path)
    rel_df = pd.read_excel(args.relative_event_path)
    data, fig_html = sync(ref_df, rel_df, args.event_mapping, args.tolerance, progress=True)
    
    with output_path.open("w") as f:
        json.dump(data, f)
    with figure_path.open("w") as f:
        f.write(fig_html)

    


        





