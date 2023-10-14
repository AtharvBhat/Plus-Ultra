# -*- coding: utf-8 -*-
"""
Script to create image dataset from TSV files containing links
"""
import logging

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from img2dataset import download  # type: ignore

from plusultra.utils import get_project_root


def create_unsplash_datset() -> None:
    """Generate unsplash dataset from tsv file"""
    logging.info("Reading Unsplash TSV File")
    unsplash_plusultra_dir = f"{get_project_root()}/data/external/unsplash"
    unsplash_dst_dir = f"{get_project_root()}/data/raw/unsplash"
    image_urls = pd.read_csv(f"{unsplash_plusultra_dir}/photos.tsv000", sep="\t")[
        "photo_image_url"
    ].to_list()
    logging.info("Done! generating list of images")
    with open(f"{unsplash_dst_dir}/urls.txt", "w", encoding="utf-8") as txt_file:
        for url in image_urls:
            txt_file.write(url + "\n")

    # download images using img2dataset
    logging.info("Done! Downloading images")
    download(
        f"{unsplash_dst_dir}/urls.txt",
        output_folder=unsplash_dst_dir,
        processes_count=6,
        resize_mode="no",
        encode_quality=100,
        encode_format="webp",
    )


@click.command()
@click.option("-d", "--dataset", default=None, type=str)
def main(dataset) -> None:
    """
    Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating Dataset for : %s", dataset)

    if dataset is None:
        raise AttributeError("No dataset argument provided")

    match dataset:
        case "unsplash":
            create_unsplash_datset()
        case _:
            raise NotImplementedError(f"{dataset} is not supported")


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()  # pylint: disable=E1120
