import pandas as pd
import tqdm
import openai
import numpy as np
import os
import pickle

from pyprojroot import here
from moral_foundations_llms import utils


api_path = os.path.join(os.environ['HOME'], 'openai/api.txt')
with open(api_path, 'r') as f:
    openai.api_key = f.read().strip()


df = pd.read_csv(here('data/aita_processed_gpt.csv'))
n_posts = df.shape[0]
labels = ['A', 'B', 'C', 'D', 'E']
log_probs = np.zeros((n_posts, len(labels)))
biases = {32: 5, 33: 5, 34: 5, 35: 5, 36: 5}
missing = []

for ii in tqdm.tqdm(range(500, 1000)):
    post = df['selftext'].iloc[ii]
    response = openai.Completion.create(
        model='text-davinci-003',
        prompt=utils.create_davinci_prompt(post),
        temperature=0.,
        logit_bias=biases,
        logprobs=5)

    log_prob_json = response['choices'][0]['logprobs']['top_logprobs'][1]
    min_log_prob = min(log_prob_json.values())

    for jj, label in enumerate(['A', 'B', 'C', 'D', 'E']):
        if label in log_prob_json:
            log_probs[ii, jj] = log_prob_json[label]
        else:
            print(f'Missing {label} in post {ii}.')
            missing.append((ii, label))
            log_probs[ii, jj] = min_log_prob


with open(here('data/davinci_results-02.pkl'), 'wb') as file:
    pickle.dump(log_probs, file)
    pickle.dump(missing, file)
