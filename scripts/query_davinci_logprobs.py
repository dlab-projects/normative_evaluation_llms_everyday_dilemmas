import pandas as pd
import tqdm
import openai
import numpy as np
import os
import pickle

from pyprojroot import here
from moral_foundations_llms import utils

# Set up OpenAI API key
api_path = os.path.join(os.environ['HOME'], 'openai/api.txt')
with open(api_path, 'r') as f:
    openai.api_key = f.read().strip()

# Read in processed data
df = pd.read_csv(here('data/aita_processed_gpt.csv'))
n_posts = df.shape[0]
model = 'text-davinci-003'
temperature = 0.
logit_bias = {32: 10, 33: 10, 34: 10, 35: 10, 36: 10}
labels = ['A', 'B', 'C', 'D', 'E']
n_logprobs = 5
verbose = True

all_probs = np.zeros((n_posts, len(labels)))

for ii in tqdm.tqdm(range(500, 3000)):
    probs = utils.get_probs(
        post=df['selftext'].iloc[ii],
        model=model,
        temperature=temperature,
        logit_bias=logit_bias,
        labels=labels,
        n_logprobs=n_logprobs,
        verbose=verbose)

    all_probs[ii] = list(probs.values())


with open(here('data/davinci_results_002.pkl'), 'wb') as file:
    pickle.dump(all_probs, file)
