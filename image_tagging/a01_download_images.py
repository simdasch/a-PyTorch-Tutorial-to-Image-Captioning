import pandas as pd
from pathlib import Path
import requests
from tqdm import tqdm
from faker import Faker


def get_image_list() -> pd.DataFrame:
    ROOT = Path(__file__).parents[1]
    image_id_file = ROOT / 'images/Image_IDs_Marken_Sortiment.csv'
    df = pd.read_csv(image_id_file)

    return df


def get_image_path(image_id: str) -> Path:
    ROOT = Path(__file__).parents[1]
    image_file = ROOT / 'images/teaser' / (image_id + '.jpg')
    image_file.parent.mkdir(parents=True, exist_ok=True)
    return image_file


def download_images():
    df = get_image_list()

    tqdm.pandas(desc="downloading ...")
    faker = Faker()

    def download_single_image(row):
        download_file = get_image_path(row['BRANDSHOPPROMOTION_IMAGE_ID'])
        download_uri = f'http://i.otto.de/i/otto/{row["BRANDSHOPPROMOTION_IMAGE_ID"]}'
        res = requests.get(download_uri, headers={
            'User-Agent': faker.user_agent()})
        if res.status_code == 200:
            download_file.write_bytes(res.content)
        else:
            print(f'Could not download image "{row["BRANDSHOPPROMOTION_IMAGE_ID"]}": {res}')


    df.progress_apply(download_single_image, axis=1)

    # for idx, row in df.iterrows():
    #     download_file = get_image_path(row['BRANDSHOPPROMOTION_IMAGE_ID'])
    #     download_uri = f'https://i.otto.de/i/otto/{row["BRANDSHOPPROMOTION_IMAGE_ID"]}'
    #     res = requests.get(download_uri)
    #     if res.status_code == '200':
    #         download_file.write_bytes(res.content)
    #     else:
    #         print(f'Could not download image "{row["BRANDSHOPPROMOTION_IMAGE_ID"]}"')

    return df


if __name__ == '__main__':
    df = download_images()