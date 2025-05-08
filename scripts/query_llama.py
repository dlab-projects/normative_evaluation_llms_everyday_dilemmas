import pandas as pd
import tqdm
import torch

from moral_foundations_llms import prompts, utils
from pyprojroot import here
from transformers import AutoTokenizer, AutoModelForCausalLM


run = 3
model_id = "meta-llama/Llama-2-7b-chat-hf"
dtype = torch.bfloat16

df = pd.read_csv(here('data/aita_final_v18.csv'))
verdicts = []
labels = []

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=dtype)

for idx in tqdm.tqdm(range(df.shape[0])):
    comment = df['selftext'].iloc[idx]
    system = prompts.aita_prompt
    chat = [
        {"role": "system", "content": system},
        {"role": "user", "content": comment},
    ]
    processed_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(processed_prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=248, do_sample=True)
    processed_output = tokenizer.decode(outputs[0])
    verdict = processed_output.split('[/INST]')[-1].strip()
    labels.append(utils.label_detector(verdict))
    verdicts.append(verdict)

out_df = pd.DataFrame(
    data={
        f'llama_label_{run}': labels,
        f'llama_label_{run}': verdicts
    }
).to_csv(f'../data/llama_run_{run}.csv', index=False)
