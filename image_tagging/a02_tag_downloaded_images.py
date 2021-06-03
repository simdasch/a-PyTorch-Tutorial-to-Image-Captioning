from pathlib import Path
import pandas as pd

from tqdm import tqdm
from image_tagging.a01_download_images import get_image_list, get_image_path
from image_tagging.tagging_from_caption import Tagger


def tag_images():
    df = get_image_list()

    tqdm.pandas(desc='tagging images ...')

    ROOT = Path(__file__).parents[1]

    model_checkpoint_file = ROOT / 'checkpoint/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
    word_map_file = ROOT / 'checkpoint/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json'

    tagger = Tagger(model_checkpoint_file=model_checkpoint_file, word_map_file=word_map_file)

    def tag_image(row):
        image_file = get_image_path(row['BRANDSHOPPROMOTION_IMAGE_ID'])
        if image_file.exists():
            return pd.Series(tagger.tag(image_file, beam_size=15))
        else:
            return pd.Series()


    result_df = df.progress_apply(tag_image, axis=1)

    result_df = result_df.join(df)

    result_file = ROOT / 'Image_IDs_Marken_Sortiment.tagged.pickle'
    result_df.to_pickle(result_file)


if __name__ == '__main__':
    tag_images()