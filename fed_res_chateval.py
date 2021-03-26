import fed
import pandas as pd
from tqdm import tqdm

# Load model
model, tokenizer = fed.load_models("microsoft/DialoGPT-large")
print("Model loaded.")

# Using this model with our chat-eval data
res_dataCSV = "../../../chatbot-eval/model_responses.csv"
res_fed_scores = "../../../chatbot-eval/res_fed_scores.tsv"

res_data = pd.read_csv(res_dataCSV, sep=",")
out_df = pd.DataFrame(res_data)
out_df["fed_scores"] = ""

# potentially delete escape characters and see if that affects the FED model?
tqdm.pandas(desc="Evaluating model responses with FED")
def fed_apply(context_str):
    return fed.evaluate(context_str, model, tokenizer)
# out_df = out_df[:20]
out_df['fed_scores'] = out_df["response_text"].progress_apply(fed_apply)
print("Scores evaluated.")

with open(res_fed_scores,'w') as write_tsv:
    write_tsv.write(out_df.to_csv(sep='\t', index=False))

print("Done.")
