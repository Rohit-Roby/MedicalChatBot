# from datasets import load_dataset
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')


# # Load the dataset
# dataset_name = "aboonaji/wiki_medical_terms_llam2_format"
# dataset = load_dataset(dataset_name)

# Convert to Pandas DataFrame (assuming we are working with the "train" split)
# df = pd.DataFrame(dataset["train"])

# Ensure the dataset has the correct text column
text_column = "text"  # Update if the dataset uses a different column name

# NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def extract_text_components(text):
    """Extracts system prompt, user instructions, and model response."""
    sys_prompt = None
    user_prompt = None
    model_resp = None

    # Extract system prompt inside <<SYS>> and <</SYS>> tags and remove the tags
    sys_match = re.search(r'<<SYS>>(.*?)<</SYS>>', text, re.DOTALL)
    if sys_match:
        sys_prompt = sys_match.group(1).strip()  # Extracted text without tags

    # Extract user prompt inside [INST] and [/INST], but remove <<SYS>> and system prompt text
    inst_match = re.search(r'\[INST\](.*?)\[/INST\]', text, re.DOTALL)
    if inst_match:
        full_inst = inst_match.group(1).strip()

        # Remove <<SYS>> section from user prompt
        user_prompt = re.sub(r'<<SYS>>.*?<</SYS>>', '', full_inst, flags=re.DOTALL).strip()

    # Extract model response outside of [INST] and [/INST] tags
    parts = re.split(r'\[/INST\]', text)
    if len(parts) > 1:
        model_resp = parts[1].strip()

    return sys_prompt, user_prompt, model_resp



def clean_user_instructions(user_inst):
    """ Remove <<SYS>> and </SYS>> tags from the user instruction text. """
    if pd.isna(user_inst):
        return ""
    user_inst = re.sub(r'<<SYS>>.*?<</SYS>>', '', user_inst, flags=re.DOTALL)  # Remove entire <</SYS>> block
    return user_inst.strip()

def preprocess_text(text):
    """ Cleans and preprocesses text. """
    if pd.isna(text):  # Handle missing values
        return ""

    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  # Stopword removal & lemmatization
    return ' '.join(tokens)