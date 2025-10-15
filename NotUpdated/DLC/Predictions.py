from typing import ClassVar
from pydantic import Field
from script2runner import CLI
from pathlib import Path


class Args(CLI):
    """
        - **Goal** : This script is intended to record predictions on a chosen video.
        - **Formats** : Predictions must be stored in a .h5 file in order to view the chosen video with the predictions. If desired, it can also be saved as a .csv file.
        - **Next step** : The .h5 file will be used in the `Visualisation_Video_Predictions.py` script.
    """
    model_path: Path = Field(..., description="Put the path to the DeepLabCut project. This corresponds to the path entered in the 'output_project_path' variable during training.", examples=["/media/filer2/T4b/UserFolders/Name/DLC-user-2025-03-13"])
    input_video_path: Path = Field(..., description="Put the path of the video you want to analyze with the prediction.", examples=['/media/filer2/T4b/Datasets/Rats/Beta-Move/myvideo.avi'])
    output_h5_path: Path = Field(..., description="The path where H5 results will be saved.", examples=["/media/filer2/T4b/UserFolders/Name/result-predict-h5.h5"], pattern="(.*\.h5$)|(^$)")
    output_csv_path: Path = Field(..., description="The path where CSV results will be saved.", examples=["/media/filer2/T4b/UserFolders/Name/result-predict-csv.csv"], pattern="(.*\.csv$)|(^$)")
    _run_info: ClassVar = dict(conda_env="dlc_py")

a = Args()

import deeplabcut
import shutil

# Ensure at least one output path is provided
if not a.output_h5_path and not a.output_csv_path:
    raise ValueError("At least one of output_h5_path or output_csv_path must be provided.")

# Create temporary output directory
analysis_output_path = Path('/media/filer2/T4b/Temporary') / f"analysis_{a.input_video_path.stem}" # à modifier pour tmp
analysis_output_path.mkdir(parents=True, exist_ok=True)

# Run DeepLabCut analysis
deeplabcut.analyze_videos(
    f'{a.model_path}/config.yaml',
    [str(a.input_video_path)],
    save_as_csv=True,
    destfolder=str(analysis_output_path)
)

# Move output files if requested
h5_file = next(analysis_output_path.glob("*.h5"), None)
csv_file = next(analysis_output_path.glob("*.csv"), None)

if a.output_h5_path and h5_file:
    shutil.move(str(h5_file), str(a.output_h5_path))
if a.output_csv_path and csv_file:
    shutil.move(str(csv_file), str(a.output_csv_path))

# Cleanup temporary directory
shutil.rmtree(analysis_output_path)

'''from typing import ClassVar
from pydantic import Field
from script2runner import CLI
from pathlib import Path
import deeplabcut
import shutil
import json
import os

class Args(CLI):
    """
        - **Goal** : This script is intended to record predictions on a chosen video.
        - **Formats** : Predictions must be stored in a .h5 file in order to view the chosen video with the predictions. If desired, it can also be saved as a .csv file.
        - **Next step** : The .h5 file will be used in the `Visualisation_Video_Predictions.py` script.
    """
    model_path: Path = Field(..., description="Put the path to the DeepLabCut project. This corresponds to the path entered in the 'output_project_path' variable during training.", examples=["/media/filer2/T4b/UserFolders/Name/DLC-user-2025-03-13"])
    input_video_path: Path = Field(..., description="Put the path of the video you want to analyze with the prediction.", examples=['/media/filer2/T4b/Datasets/Rats/Beta-Move/myvideo.avi'])
    output_h5_path: Path = Field(..., description="The path where H5 results will be saved.", examples=["/media/filer2/T4b/UserFolders/Name/result-predict-h5.h5"], pattern="(.*\.h5$)|(^$)")
    output_csv_path: Path = Field(..., description="The path where CSV results will be saved.", examples=["/media/filer2/T4b/UserFolders/Name/result-predict-csv.csv"], pattern="(.*\.csv$)|(^$)")
    _run_info: ClassVar = dict(conda_env="dlc_py")

a = Args()

# Ensure at least one output path is provided
if not a.output_h5_path and not a.output_csv_path:
    raise ValueError("At least one of output_h5_path or output_csv_path must be provided.")

# Create temporary output directory
analysis_output_path = Path('/media/filer2/T4b/Temporary') / f"analysis_{a.input_video_path.stem}"
analysis_output_path.mkdir(parents=True, exist_ok=True)

status = {"status": "failed"}  # default

try:
    # Run DeepLabCut analysis
    deeplabcut.analyze_videos(
        str(a.model_path / "config.yaml"),
        [str(a.input_video_path)],
        save_as_csv=True,
        destfolder=str(analysis_output_path)
    )

    # Move output files if requested
    h5_file = next(analysis_output_path.glob("*.h5"), None)
    csv_file = next(analysis_output_path.glob("*.csv"), None)

    if a.output_h5_path and h5_file:
        a.output_h5_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(h5_file), str(a.output_h5_path))
    if a.output_csv_path and csv_file:
        a.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(csv_file), str(a.output_csv_path))

    status = {"status": "done"}

except Exception as e:
    status["message"] = str(e)

finally:
    # Write status.json next to the output files (based on output_csv_path)
    output_status_dir = a.output_csv_path.parent if a.output_csv_path else a.output_h5_path.parent
    output_status_dir.mkdir(parents=True, exist_ok=True)
    with open(output_status_dir / "status.json", "w") as f:
        json.dump(status, f)

    # Clean up temp folder only if success
    if status["status"] == "done" and analysis_output_path.exists():
        shutil.rmtree(analysis_output_path)

    print(f"✅ Fin du script Predictions.py, statut = {status['status']}")
'''