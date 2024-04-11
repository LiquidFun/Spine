import contextlib
import io
import os
import shutil
import socket
import sys

import mlflow
import yaml
from pytorch_lightning.cli import LightningCLI


def colored_tracebacks():
    from rich import traceback

    traceback.install(width=120)


def setup_mlflow():
    from spine_segmentation.resources.preloaded import get_secret

    os.environ["MLFLOW_TRACKING_USERNAME"] = get_secret("mlflow_username")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = get_secret("mlflow_password")

    mlflow.set_tracking_uri(uri="<mlflow-url>")

    possible_config_paths = [arg for arg in sys.argv if arg.startswith("--config=")]
    if len(possible_config_paths) > 0:
        config_path = possible_config_paths[0].split("--config=")[1]

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            if "experiment_name" in config["model"]["init_args"]:
                experiment_name = config["model"]["init_args"]["experiment_name"]
                mlflow.set_experiment(experiment_name)
            else:
                mlflow.set_experiment("spine_annotator")

        copy_to_path = "/tmp/config.yaml"
        shutil.copy(config_path, copy_to_path)
        mlflow.set_tag("ConfigPath", str(config_path))
        mlflow.log_artifact(copy_to_path, "config")
    else:
        mlflow.set_experiment("spine_annotator")

    mlflow.set_tag("mlflow.runName", "debug_run")
    mlflow.set_tag("Hostname", socket.gethostname())

    # mlflow.log_artifact("config.yaml")
    mlflow.pytorch.autolog(log_models=False, log_datasets=False)

    run_cli_but_only_log_config_in_mlflow()


def run_cli_but_only_log_config_in_mlflow():
    string_io = io.StringIO()
    with contextlib.redirect_stdout(string_io):
        try:
            sys.argv.append("--print_config")
            LightningCLI()
        except:
            pass
        finally:
            sys.argv.pop()
    config = string_io.getvalue()
    with open("/tmp/config_printed.yaml", "w") as file:
        file.write(config)
    mlflow.log_artifact("/tmp/config_printed.yaml", "config")


def main():
    colored_tracebacks()
    # setup_mlflow()
    LightningCLI()


if __name__ == "__main__":
    main()
