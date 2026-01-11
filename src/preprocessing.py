import pandas as pd
import re

def clean_lyrics(lyrics):
    if not isinstance(lyrics, str): return ''
    lyrics = re.sub(r'\n', ' ', lyrics)
    lyrics = re.sub(r'\s+', ' ', lyrics)
    lyrics = re.sub(r'[^A-Za-z0-9\u0980-\u09FF ]+', '', lyrics)
    return lyrics

def load_data(path1, path2):
    df1 = pd.read_csv(path1, encoding='utf-8')
    df2 = pd.read_csv(path2, encoding='utf-8')
    hybrid = pd.concat([
        df1[['Song Title', 'Lyrics']].rename(columns={'Lyrics': 'lyrics', 'Song Title': 'title'}),
        df2[['title', 'lyrics']]
    ], ignore_index=True)
    hybrid['cleaned'] = hybrid['lyrics'].apply(clean_lyrics)
    return hybrid
