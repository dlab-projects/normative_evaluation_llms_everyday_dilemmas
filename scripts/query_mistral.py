import pandas as pd
import tqdm
import torch

from moral_foundations_llms import prompts, utils
from pyprojroot import here
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "mistralai/Mistral-7B-Instruct-v0.2"
dtype = torch.bfloat16

df = pd.read_csv(here('data/aita_final_v4.csv'))
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
    prompt = prompts.single_prompt(
        comment=comment,
        system=prompts.aita_base_system_prompt)
    chat = [
        { "role": "user", "content": prompt},
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
        'mistral_label_2': labels,
        'mistral_verdict_2': verdicts
    }
).to_csv('../data/mistral_run_2.csv', index=False)
