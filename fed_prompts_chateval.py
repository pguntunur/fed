import fed
import pandas as pd
from tqdm import tqdm

# Load model
model, tokenizer = fed.load_models("microsoft/DialoGPT-large")
print("Model loaded.")

# Using this model with our chat-eval data
prompt_dataCSV = "../../../chatbot-eval/turksData/prompts.csv"
prompt_fed_scores = "../../../chatbot-eval/turksData/prompt_fed_scores.tsv"

prompt_data = pd.read_csv(prompt_dataCSV, sep=",")
out_df = pd.DataFrame(prompt_data)
out_df["fed_scores"] = ""

# potentially delete escape characters and see if that affects the FED model?
tqdm.pandas(desc="Evaluating with FED")
def fed_apply(context_str):
    return fed.evaluate(context_str, model, tokenizer)

out_df['fed_scores'] = out_df["prompt_text"].progress_apply(fed_apply)
print("Scores evaluated.")

with open(tc_fed_scores,'w') as write_tsv:
    write_tsv.write(out_df.to_csv(sep='\t', index=False))

print("Done.")
