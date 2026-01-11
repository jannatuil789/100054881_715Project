import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_merge_data(path1, path2):
    df_all_songs = pd.read_csv(path1, encoding='utf-8')
    df_bangla_lyrics = pd.read_csv(path2, encoding='utf-8')
    
    hybrid_dataset = pd.concat([
        df_all_songs[['Song Title', 'Lyrics']].rename(columns={'Lyrics': 'lyrics', 'Song Title': 'title'}),
        df_bangla_lyrics[['title', 'lyrics']]
    ], ignore_index=True)
    return hybrid_dataset

def clean_lyrics(lyrics):
    if not isinstance(lyrics, str):
        return ''
    lyrics = re.sub(r'\n', ' ', lyrics)
    lyrics = re.sub(r'\s+', ' ', lyrics)
    # Keep English and Bengali characters
    lyrics = re.sub(r'[^A-Za-z0-9\u0980-\u09FF ]+', '', lyrics)
    return lyrics

def get_tfidf_features(texts, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    return vectorizer.fit_transform(texts), vectorizer
