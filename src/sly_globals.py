import os
import sys
import time
from pathlib import Path

import supervisely as sly
from dotenv import load_dotenv

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

app_root_directory = str(Path(__file__).parent.absolute().parents[0])
sly.logger.info(f"Root source directory: {app_root_directory}")
project_dir = os.path.join(app_root_directory, "sly_project")
data_dir = os.path.join(app_root_directory, "data")
checkpoints_dir = os.path.join(data_dir, "checkpoints")
os.environ["SLY_APP_DATA_DIR"] = data_dir
Path(checkpoints_dir).mkdir(exist_ok=True, parents=True)

api = sly.Api()
project_fs: sly.Project = None
project_id = sly.env.project_id()
project_info = api.project.get_info_by_id(project_id)
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
workspace = api.workspace.get_info_by_id(project_info.workspace_id)
# team.id will be used for storaging app results in team files
team = api.team.get_info_by_id(workspace.team_id)

COLUMNS_COUNT = 6
PREVIEW_IMAGES_COUNT = 18
