from pathlib import Path
import pandas as pd
import numpy as np
import jinja2

from image_tagging.tagging_from_caption import get_non_wanted_tags

ROOT = Path(__file__).parents[1]

def get_result_df():
    # result_file = ROOT / 'Image_IDs_Marken_Sortiment.tagged.sample.pickle'
    result_file = ROOT / 'Image_IDs_Marken_Sortiment.tagged.pickle'

    df = pd.read_pickle(result_file)
    df = df.assign(image_name=df['BRANDSHOPPROMOTION_IMAGE_ID'])
    df = df.assign(image_url='https://i.otto.de/i/otto/' + df['image_name'] + '?maxW=300&maxH=300&sm=C&fmt=jpg')
    df = df.assign(index=df.index)

    # remove non downloadable image
    df = df[~df['score'].isnull()]
    return df


template = jinja2.Template('''
<table width="100%" style="background: transparent; border-bottom: 1px solid lightgray;">
    <tr style="background: transparent;">
        <td width="40%" style="vertical-align: top;">
            <div align="center"><strong>Image #{{ index }}</strong></div>
            <img src="{{ image_url }}" style="width: 1000px; position absolute; z-index: 1;">
            <div style="text-align: center; padding: 15px 0;">
                <a href="https://i.otto.de/i/otto/{{ image_name }}" target="_{{ image_name }}">{{ image_name }}</a>
            </div>
        </td>
        <td width="35%" style="vertical-align: top;">
            <div align="center"><strong>Tags</strong></div>
            {% if tags %}
            <table style="width: 100%; margin-bottom: 15px;">
                <tr>
                    <th>tag</th>
                    <th>word score</th>
                </tr>
                {% for tag in tags %}
                <tr style="background: transparent;">
                    <td style="width: 30%;">{{ tag['name'] }}</td>
                    <td style="width: 70%; position: relative;">
                        <div style="background: lightgray; position: relative; width: 100%; height: 15px">
                            <div style="background: gray; padding-right: 2px; padding-top: 1px; color: #fff; font-size: 10px; position: absolute; left: 0; top: 0; height: 100%; width: {{ tag['score_word'] * 100 }}%;">
                                {{ "%.1f%%"|format(tag['score_word']*100|float) }}
                            </div>
                        </div>
                    </td>
                </tr>
                {% endfor %}
            </table>
            {% endif %}
        </td>
    </tr>
</table>
''')


def render(row):
    tags_dict = []
    non_wanted = get_non_wanted_tags()
    tag_list = row['score']
    tag_list = sorted(tag_list, key=lambda t: t.word_score, reverse=True)
    for tag in tag_list:
        if tag.tag not in non_wanted:
            tags_dict.append({'name': tag.tag, 'score_caption': tag.caption_score_compensated, 'score_word': tag.word_score})
    output = template.render(
        index=row['index'],
        image_name=row['image_name'],
        image_url=row['image_url'],
        tags=tags_dict,
        description_tags=False
    )
    return output


def render_dataframe(df: pd.DataFrame, part: int, of: int):
    print(f'rendering result {part:03d} of {of:03d}')
    if part == 17:
        a=42

    rows = df.apply(render, axis=1)

    contents = '\n\n'.join(rows)

    html = f"""
<!DOCTYPE html>
<html>
<head>
<title>Page Title</title>
</head>
<body>

{contents}

</body>
</html>"""

    output_file = ROOT / 'image_tagging/results' / f'results_{part:03d}_of_{of:03d}.html'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html)


def visualize():
    df = get_result_df()

    chunksize = 40
    chunk_count = len(df) // chunksize + 1

    for i, chunk in enumerate(np.array_split(df, chunk_count), 1):
        render_dataframe(chunk, i, chunk_count)


def generate_result_csv():
    df = get_result_df()

    def get_result_cols(row):
        non_wanted = get_non_wanted_tags()
        tag_list = row['score']
        tag_list = list(filter(lambda x: x.tag not in non_wanted, tag_list))
        tag_list = sorted(tag_list, key=lambda t: t.word_score, reverse=True)
        tags = '|'.join([t.tag for t in tag_list])
        confidences = '|'.join([str(t.word_score) for t in tag_list])
        return pd.Series({'tags': tags, 'confidences': confidences})

    df = df.apply(get_result_cols, axis=1).join(df)

    df = df[['image_name', 'tags', 'confidences']]

    output_file = ROOT / 'image_tagging/results' / f'results.csv'
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    # df = visualize()
    generate_result_csv()
    # a = 42