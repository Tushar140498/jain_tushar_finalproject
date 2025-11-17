"""
data_prep.py
-- MAIN Work of this python file is --
It loads the raw SMS Spam dataset (SMSSpamCollection) from data folder
, cleans it, and saves a cleaned CSV at data/sms_spam.csv.


"""

import os
import pandas as pd
import re

DATA_DIRECTORY = "data"
RAW_FILE = os.path.join(DATA_DIRECTORY, "SMSSpamCollection")
OUTPUT_FILE = os.path.join(DATA_DIRECTORY, "sms_spam.csv")

os.makedirs(DATA_DIRECTORY, exist_ok=True)


# Function for text cleaning

def clean_text(text):
    text = str(text).lower()                              # lowercase
    text = re.sub(r"http\S+|www\S+", " ", text)           # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)              # remove punctuation/symbols
    text = re.sub(r"\s+", " ", text).strip()              # normalize spaces
    return text


# Loading Raw Dataset

def load_raw_dataset():

    df = pd.read_csv(
        RAW_FILE,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8"
    )
    
    # Convert labels: ham → 0, spam → 1
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df


# Main function

def main():
    
    df = load_raw_dataset()

    df["text_clean"] = df["text"].apply(clean_text)
    
    df.to_csv(OUTPUT_FILE, index=False)

    print("Your data is prepared !")


if __name__ == "__main__":
    main()
