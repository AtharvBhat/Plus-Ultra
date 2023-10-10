# Data Sources

### Super-resolution and Image Enhancement Data: 

Models trained to perform Superresolution and Image Enhancement tasks use the [unsplash](https://unsplash.com/data) lite data. <br>
The unsplash data is packaged into `.tsv` files, our of which the `photos.tsv000` contains image metadata along with their permenant links. <br>
This repo does not come with the data but, scripts are provided to prepare the dataset.

To prepare the dataset, download and save `photos.tsv000` from unsplash dataset in `data/external/unsplash` and run `src/data/make_dataset.py --dataset unsplash`