# import fed

# Load model
model, tokenizer = fed.load_models("microsoft/DialoGPT-large")

# # Evaluate
# conversation = "<|endoftext|> Hi! <|endoftext|> Hello, how is your day? <|endoftext|> It's good. It's raining a bit, but I am enjoying a good book. How about you? <|endoftext|> It's good, I just got back from walking my dog What book did you read?"
#
#
# scores = fed.evaluate(conversation,
#                       model,
#                       tokenizer)
#
# print(scores)

# Using this model with our chat-eval data

import pandas as pd

tc_usr_dataTSV = "../../../chatbot-eval/tc_usr_data.tsv"
tc_fed_scores = "../../../chatbot-eval/tc_fed_scores.tsv"

# tc_usr_dataTSV = "../chatbot-eval/tc_usr_data.tsv"
# tc_fed_scores = "../chatbot-eval/tc_fed_scores.tsv"

tc_usr_dataTSV_r = pd.read_csv(tc_usr_dataTSV, sep="\t")
out_df = pd.DataFrame(tc_usr_dataTSV_r["context"])
out_df["fed_scores"] = ""

# Change text format of chateval dataset according to input in fed.py
for i, row in out_df.iterrows():
    out_df.at[i,'fed_scores'] = fed.evaluate(row["context"], model, tokenizer)

with open(tc_fed_scores,'w') as write_tsv:
    write_tsv.write(out_df.to_csv(sep='\t', index=False))
