from datasets import load_dataset
import pandas as pd
from textprocessingfunction import preprocess_text, extract_text_components, clean_user_instructions

# Load the dataset
dataset_name = "aboonaji/wiki_medical_terms_llam2_format"
dataset = load_dataset(dataset_name)
df = pd.DataFrame(dataset["train"])

text_column = "text"
# Apply extraction to the dataset
df[['sys_prompt', 'user_prompt', 'model_resp']] = df[text_column].apply(
    lambda x: pd.Series(extract_text_components(x))
)

# Clean user instructions by removing the <<SYS>> and </SYS>> tags
df['user_prompt'] = df['user_prompt'].apply(clean_user_instructions)
df['sys_prompt'] = df['sys_prompt'].apply(clean_user_instructions)
# Preprocess extracted text fields
df['user_prompt'] = df['user_prompt'].apply(preprocess_text)
df['sys_prompt'] = df['sys_prompt'].apply(preprocess_text)
df['model_resp'] = df['model_resp'].apply(preprocess_text)

# Drop original text column
df = df.drop(columns=["text"])
df.to_csv('processed_data.csv', index=False)
print(df.head(2))
