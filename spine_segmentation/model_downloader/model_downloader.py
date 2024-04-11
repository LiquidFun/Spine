import os
from typing import Optional
from pathlib import Path

import requests
from tqdm import tqdm


def _get_config_path(program_name="spine") -> Path:
    """Get OS-agnostic path for program configuration."""

    home = Path.home().absolute()

    if os.name == "posix":
        # Linux, macOS, and other UNIX-like systems
        config_path = home / ".config" / program_name
    elif os.name == "nt":
        # Windows
        local_app_data = Path(os.getenv("LOCALAPPDATA"))  # For machine-local configurations
        config_path = local_app_data / program_name
    else:
        # Fallback for unknown OS
        config_path = home / f".{program_name}"

    config_path.mkdir(parents=True, exist_ok=True)
    return config_path


def download_file_with_progress(url: str, local_filename: Optional[Path] = None) -> Path:
    """Download file if it does not exist, and save persistently to local file. Return its path either way."""
    if local_filename is None:
        local_filename = _get_config_path() / '/'.join(url.split("/")[-2:])

    if local_filename.exists():
        return local_filename

    local_filename.parent.mkdir(exist_ok=True, parents=True)

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))

    block_size = 1024

    print(f"Downloading {local_filename}...")
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(local_filename, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")
    else:
        print(f"Downloaded {local_filename}")

    return local_filename


def main():
    # Example usage
    url = "https://github.com/LiquidFun/Spine/releases/download/onnx_models_1.0.0/2023-11-06_Seg2_Unet_resnet152_896px.onnx"
    download_file_with_progress(url)


if __name__ == "__main__":
    main()
