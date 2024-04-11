from pathlib import Path

PROJECT_PATH = Path(__file__).absolute().parent.parent.parent

DATASETS_PATH = PROJECT_PATH / "datasets"
REPORTS_PATH = PROJECT_PATH / "reports"
MODELS_PATH = PROJECT_PATH / "models"
SECRETS_PATH = PROJECT_PATH / "secrets"
LOGS_PATH = PROJECT_PATH / "logs"

PLOTS_PATH = REPORTS_PATH / "plots"
RESULTS_PATH = REPORTS_PATH / "results"

MEASURE_STATISTICS_PATH = DATASETS_PATH / "measure_statistics_new.csv"
VECTOR_TABLE_PATH = DATASETS_PATH / "vector_table_new.npz"

# Paths outside repository

CACHE_PATH = Path("/tmp/spine-segmentation-cache")
CACHE_PATH.mkdir(exist_ok=True)

NAS_RESEARCH_PATH = Path("~/devel/data_remote/nas_research").expanduser().resolve()
NAKO_DATASET_PATH = NAS_RESEARCH_PATH / "path"

LOCAL_NAKO_DATASET_PATH = Path.home() / "Data/nako/2022-08-22"

VAL_SPLIT_CSV_PATH = DATASETS_PATH / "nako_splits/val_seg.csv"
TRAIN_SPLIT_CSV_PATH = DATASETS_PATH / "nako_splits/train_seg.csv"
