import os
from pathlib import Path

import supervisely as sly
from dotenv import load_dotenv

load_dotenv("local.env")
load_dotenv(os.path.expanduser("~/supervisely.env"))

app_root_directory = str(Path(__file__).parent.absolute().parents[0])
sly.logger.info(f"Root source directory: {app_root_directory}")
# projects_dir = os.path.join(app_root_directory, "sly_projects")
# project_dir = None

api = sly.Api.from_env()
project_fs: sly.Project = None
project_id = sly.env.project_id(raise_not_found=False)
dataset_id = sly.env.dataset_id(raise_not_found=False)
project_info = api.project.get_info_by_id(project_id)
project_meta = sly.ProjectMeta.from_json(api.project.get_meta(project_id))
workspace = api.workspace.get_info_by_id(project_info.workspace_id)
# team.id will be used for storaging app results in team files
team = api.team.get_info_by_id(workspace.team_id)
# os.environ["TEAM_ID"] = str(team.id)

DATASET_IDS = [dataset_id] if dataset_id else []
COLUMNS_COUNT = 6
PREVIEW_IMAGES_COUNT = 18
