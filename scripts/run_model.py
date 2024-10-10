import numpy as np
import pandas as pd
import torch

from moral_foundations_llms.models import RedditScenarioModel
from moral_foundations_llms.loaders import AITACorpus

from torch.utils.data import DataLoader
from tqdm import tqdm

CHECKPOINT_PATH = "checkpoints/checkpoint_Social_Norms_epoch=1-val_loss=0.25.ckpt"
TOKENIZER = 'roberta-large'
AXIS = 'social_norms'

corpus = AITACorpus('data/aita_final_v11.csv', tokenizer=TOKENIZER)
model = RedditScenarioModel.load_from_checkpoint(CHECKPOINT_PATH).to('cuda:0')
loader = DataLoader(corpus, batch_size=32, shuffle=False)

model.eval()
predictions = []
with torch.no_grad():
    for batch in tqdm(loader):
        input_ids = batch['input_ids'].to('cuda:0')
        attention_mask = batch['attention_mask'].to('cuda:0')
        outputs = model(input_ids, attention_mask)
        probs = torch.sigmoid(outputs).to('cpu').numpy()
        predictions.append(probs)

predictions = np.concatenate(predictions)
df = pd.read_csv('data/aita_final_v11.csv')
df[f'{AXIS}_prob'] = predictions
print((df[f'{AXIS}_prob'] > 0.5).mean())
df.to_csv('data/aita_final_v12.csv', index=False)
