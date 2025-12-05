import pandas as pd
pd.set_option("display.max_columns", None)
import numpy as np
from tensorflow.keras.models import load_model
from lib_file import lib_path
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import torch
# from lib_file import lib_path
from transformers import GPT2Model, GPT2Tokenizer
from transformers import pipeline
from urllib.parse import urlparse
import googleapiclient
from langdetect import detect
from IPython import display


model = load_model("models/ConvolutionalGatedRecurrentUnit_model.h5", compile=False)
gpt2_model = GPT2Model.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
pipeline_ = pipeline("sentiment-analysis")
display.clear_output()


class_labels = ['NEGATIVE', 'POSITIVE']


api_key = "AIzaSyB1uysE-Mn5kUml8hxafw9_TVQICQPfoMI"
youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)


def get_video_id_from_url(url):
    parsed_url = urlparse(url)
    if parsed_url.netloc == 'www.youtube.com' or parsed_url.netloc == 'youtube.com':
        query_params = parsed_url.query
        query_params = query_params.split('&')
        for param in query_params:
            key_value = param.split('=')
            if key_value[0] == 'v':
                return key_value[1]
    elif parsed_url.netloc == 'youtu.be':
        path_segments = parsed_url.path.split('/')
        if len(path_segments) > 1:
            return path_segments[1]

    return None


def retrieve_all_comments(video_id):
    comments = []
    next_page_token = None

    while True:
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            order='time',
            pageToken=next_page_token
        ).execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)

        if 'nextPageToken' in response:
            next_page_token = response['nextPageToken']
        else:
            break

    return comments


def is_english_sentence(sentence):
    try:
        lang = detect(sentence)
        if lang == 'en':
            return True
        else:
            return False
    except:
        return False
    

def text_cleaning(text):
    text = text.lower()
    text = BeautifulSoup(text, "html.parser")
    text = text.get_text()
    emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # Emojis
                            u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # Transport & map symbols
                            u"\U0001F700-\U0001F77F"  # Alphabetic presentation forms
                            u"\U0001F780-\U0001F7FF"  # Geometric shapes
                            u"\U0001F800-\U0001F8FF"  # Miscellaneous symbols"
                            u"\U0001F900-\U0001F9FF"  # Supplemental symbols & pictographs
                            u"\U0001FA00-\U0001FA6F"  # Extended-A
                            u"\U0001FA70-\U0001FAFF"  # Extended-B
                            u"\U0001F004-\U0001F0CF"  # Mahjong tiles
                            u"\U0001F170-\U0001F251"  # Enclosed characters
                            u"\U00020000-\U0002F73F"  # Chinese, Japanese, and Korean characters
                            u"\U000E0000-\U000E007F"  # Tags
                            "]+", flags=re.UNICODE)
    text = re.sub(emoji_pattern, '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


# video_url = input("Paste youtube's appropriate link...\n")

def youtubeCommentsAnalysis(video_url):
    video_id = get_video_id_from_url(video_url)
    if video_id is not None:
        print(f"The video ID, {video_id}, has been detected.")
    else:
        print("The video ID could not be retrieved from the URL. Please try a different URL.")


    all_comments = retrieve_all_comments(video_id)
    del(all_comments[0])
    all_comments[:10]


    english_comments = []
    for sent in tqdm(all_comments):
        result = is_english_sentence(sent)
        if result:
            english_comments.append(sent)
        else:
            continue


    english_comments[:10]


    cleaned_sentences = []
    for sentence in tqdm(english_comments):
        cleaned_text = text_cleaning(sentence)
        cleaned_sentences.append(cleaned_text)

    df = pd.DataFrame(data={"Text": cleaned_sentences})
    df['length']=df['Text'].apply(lambda x: len(x.split()))
    df = df.loc[(df['length'] <= 500) & (df['length'] >= 2)]
    df = df.drop(labels='length', axis=1)
    df.head(10)



    results = []
    for text in tqdm(df['Text'].values.tolist()):
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = gpt2_model(**inputs)
            features_ = pipeline_(text)
            features_res = output.last_hidden_state.mean(dim=1).squeeze().tolist()
            features_res = np.reshape(np.array(features_res), (1, len(features_res), 1))
            features_res = model.predict(features_res, verbose=0)
            results.append(features_[0]['label'])


    df['sentiment'] = results
    positive_df = df.loc[df['sentiment']=='POSITIVE']
    positive_df.to_csv("static/result/positive_df.csv", index=False)

    negative_df = df.loc[df['sentiment']=='NEGATIVE']
    negative_df.to_csv("static/result/negative_df.csv", index=False)

    total_sample_size = df.shape[0]
    positive_sample_size = positive_df.shape[0]
    negative_sample_size = negative_df.shape[0]

    print(f"Total useful english comments: {df.shape[0]}")
    print(f"Posititive comments: {df.loc[df['sentiment']=='POSITIVE'].shape[0]}")
    print(f"Negative comments: {df.loc[df['sentiment']=='NEGATIVE'].shape[0]}")

    return total_sample_size, positive_sample_size, negative_sample_size