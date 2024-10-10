import numpy as np
import pandas as pd
import torch

from moral_foundations_llms.models import RedditScenarioModel
from moral_foundations_llms.loaders import AITAVerdictCorpus

from torch.utils.data import DataLoader
from tqdm import tqdm

AITA_PATH = "data/aita_final_v12.csv"
TOKENIZER = 'roberta-large'
AXES = [
    'fairness',
    'feelings',
    'harms',
    'honesty',
    'relational_obligation',
    'social_norms']
REASONS = [
    'top_comment',
    'gpt3.5_reason_1',
    'gpt4_reason',
    'claude_reason',
    'bison_reason_1',
    'gemma_reason_1',
    'mistral_reason_1',
    'llama_reason_1']
df = pd.read_csv(AITA_PATH)


for axis in AXES:
    for reason in REASONS:
        corpus = AITAVerdictCorpus(
            path=AITA_PATH,
            tokenizer=TOKENIZER,
            reason=reason)
        checkpoint_path = f"checkpoints/aita_{axis}_model.ckpt"
        model = RedditScenarioModel.load_from_checkpoint(checkpoint_path).to('cuda:0')
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
        NEW_COL = f"{reason}_{axis}_prob"
        df[NEW_COL] = predictions
        print(f"{NEW_COL}: {(df[NEW_COL] > 0.5).mean()}")

df.to_csv(AITA_PATH, index=False)
