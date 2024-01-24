import pandas as pd
import tqdm
import openai
import nltk
import numpy as np
import os
import pickle
import sys

from pyprojroot import here
from moral_foundations_llms import utils

API_PATH = os.path.join(os.environ['HOME'], 'openai/api.txt')
IN_FILE = here('data/aita_comments_Aug17_labels_update_Aug31.csv')
OUT_FILE = here('data/aita_nan_comments_labels.csv')
FAIL_FILE = here('data/comment_labels_fail.pkl')
# Read in API key
with open(API_PATH, 'r') as f:
    openai.api_key = f.read().strip()
# Read in processed data
df = pd.read_csv(IN_FILE)
out = pd.read_csv(OUT_FILE)
df = df[df['comment_author'] != 'AutoModerator']
df = df[df['comment_label'].isna()].reset_index(drop=True)

n_query = df.shape[0]
failed = []
responses = {}

posts = np.argwhere(out['gpt_comment_label'].isna()).ravel()

# Iterate over posts
for post in tqdm.tqdm(posts):
    retries = 0
    completed = False
    while not completed and retries < 3:
        # Run GPT query
        # print(f'Post {post}, retry {retries}')
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[
                    {"role": "system", "content": utils.comment_label_prompt},
                    {"role": "user", "content": df['comment'].iloc[post]}
                ],
                temperature=0.4)
        # Handle exceptions
        except openai.error.Timeout as e:
            print(f"Post {post}, OpenAI API returned a Timeout Error: {e}")
            retries += 1
            continue
        except openai.error.APIError as e:
            print(f"Post {post}, OpenAI API returned an API Error: {e}")
            retries += 1
            continue
        except openai.error.APIConnectionError as e:
            print(f"Post {post}, OpenAI API request failed to connect: {e}")
            retries += 1
            continue
        except openai.error.ServiceUnavailableError as e:
            print(f"Post {post}, OpenAI API returned a Service Unavailable Error: {e}")
            retries += 1
            continue
        # Extract answer
        answer = response['choices'][0]['message']['content']
        # Store answer, dilemma, label, and reason
        responses[post] = answer
        # Place in dataframe
        out.loc[post, 'gpt_comment_label'] = answer
        completed = True
    if not completed:
        failed.append(post)

out.to_csv(OUT_FILE, index=False)

with open(FAIL_FILE, 'wb') as file:
    pickle.dump([failed, responses], file)