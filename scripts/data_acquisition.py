"""
Downloads and unpacks data.
"""
from pathlib import Path

import gdown
import os
import tarfile


def get_data(base_path: str) -> None:
    """
    Downloads and unpacks dta. Does nothing if already available.
    :param base_path:
    """

    archive_path = os.path.join(base_path, "l2rpn_data.tgz")
    extraction_indicator_path = os.path.join(base_path, "data/extract_data_here")
    is_archive_available = os.path.exists(archive_path)
    is_unpacked_data_available = all(
        os.path.exists(os.path.join(base_path, dir_name))
        for dir_name in ("l2rpn_case14_sandbox", "l2rpn_wcci_2020", "rte_case5_example")
    )

    # Download.
    if not is_archive_available and not is_unpacked_data_available:
        print("Downloading data.")
        gdown.download(
            'https://drive.google.com/uc?id=15oW1Wq7d6cu6EFS2P7A0cRhyv8u_UqWA',
            archive_path,
            quiet=False
        )

    # Unpack.
    if not is_unpacked_data_available:
        print("Unpacking data.")
        tar = tarfile.open(archive_path, "r:gz")
        tar.extractall(path=Path(base_path).parent.absolute())
        tar.close()

    # Delete archive and indicator file.
    if is_archive_available:
        os.remove(archive_path)
    if os.path.exists(extraction_indicator_path):
        os.remove(extraction_indicator_path)


if __name__ == '__main__':
    get_data(base_path="maze_smaac/data/")
