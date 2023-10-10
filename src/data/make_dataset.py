# -*- coding: utf-8 -*-
"""
Script to create image dataset from TSV files containing links
"""
import logging
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from img2dataset import download

from src.utils import get_project_root


def create_unsplash_datset() -> None:
    """Generate unsplash dataset from tsv file"""
    logging.info("Reading Unsplash TSV File")
    unsplash_src_dir = f"{get_project_root()}/data/external/unsplash"
    unsplash_dst_dir = f"{get_project_root()}/data/raw/unsplash"
    image_urls = pd.read_csv(f"{unsplash_src_dir}/photos.tsv000", sep="\t")[
        "photo_image_url"
    ].to_list()
    logging.info("Done! generating list of images")
    with open(f"{unsplash_dst_dir}/urls.txt", "w") as txt_file:
        for url in image_urls:
            txt_file.write(url + "\n")

    # download images using img2dataset
    logging.info("Done! Downloading images")
    download(
        f"{unsplash_dst_dir}/urls.txt",
        output_folder=unsplash_dst_dir,
        processes_count=6,
        resize_mode="no",
        encode_quality=9,
        encode_format="png",
    )


@click.command()
@click.option("-d", "--dataset", default=None, type=str)
def main(dataset) -> None:
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Creating Dataset for : {dataset}")

    if dataset is None:
        raise (AttributeError("No dataset argument provided"))

    match dataset:
        case "unsplash":
            create_unsplash_datset()
        case _:
            raise (NotImplementedError(f"{dataset} is not supported"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
