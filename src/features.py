import re
import random
import math
import csv
import pandas as pd
from datetime import datetime
from itertools import combinations
import os
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.paths_config import PROCESSED_DIR, MODEL_DIR

print(PROCESSED_DIR)
# Original intents dictionary
intents = {
    "Product Inquiry": [
        "What are the features of this laptop?",
        "Is this phone available?",
        "What is the price of the new headphones?",
        "Do you have this product in stock?",
        "Expected availability date for the product",
        "What are the different color options that are available for the product?",
        "Help me with the products that have discounts",
    ],
    "Order Tracking": [
        "Where is my order?",
        "How long will delivery take?",
        "Can you provide the tracking details?",
        "I want to check the status of my shipment.",
        "There is delay in the order delivery, can you please let me know the reason",
        "System shows that order is delivered but I have not received any order",
        "I've been waiting for the order long time"
    ],
    "Refund Request": [
        "How do I get a refund?",
        "Can I return my order?",
        "What is the process for a refund?",
        "Can I cancel my order and get a refund?",
        "It's been long time since I have raised the refund, but amount is not credited",
        "When can I expect the refund to be processed",
        "I don't want this product anymore",
        "Product I received is different from the one that I placed order, need help with refund",
    ],
    "Store Policy": [
        "What is your return policy?",
        "Do you offer free shipping?",
        "Can you explain your warranty terms?",
        "What are the delivery charges?",
        "What are the options for free delivery",
    ],
}

# Simple paraphrase templates
paraphrase_templates = [
    "Can you tell me {}?",
    "I need info on {}.",
    "Could you explain {}?",
    "I'm looking to know {}.",
    "Please help me understand {}.",
    "Would like to get details about {}.",
    "{} please?",
    "Need help with {}.",
]

# Synonym dictionary
synonyms = {
    "product": ["item", "goods"],
    "refund": ["repayment", "money back"],
    "order": ["purchase", "shipment"],
    "delivery": ["shipping", "dispatch"],
    "available": ["in stock", "in store"],
    "features": ["specs", "specifications"],
    "price": ["cost", "rate"]
}

def augment_phrase(phrase):
    modified = phrase
    for key, syns in synonyms.items():
        if key in modified.lower():
            replacement = random.choice(syns)
            modified = modified.replace(key, replacement)
    template = random.choice(paraphrase_templates)
    return template.format(modified.lower().capitalize())

def generate_multi_intent_data(intents_dict, total_samples=100, max_combination_size=2):
    intent_names = list(intents_dict.keys())
    synthetic_data = []

    while len(synthetic_data) < total_samples:
        # Randomly select how many intents to combine
        k = random.randint(2, max_combination_size)
        intent_combo = random.sample(intent_names, k)

        queries = []
        for intent in intent_combo:
            phrase = random.choice(intents_dict[intent])
            if random.random() < 0.7:
                phrase = augment_phrase(phrase)
            queries.append(phrase)

        full_query = " ".join(queries)
        synthetic_data.append((full_query.strip(), intent_combo))

    return synthetic_data

def save_to_csv(data, filename="multi_intent.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "intents"])
        for text, intents in data:
            text = preprocess_query(text)
            writer.writerow([text, intents])
    print(f"Data saved to {filename}")

def expand_contractions(text):
    contractions_dict = {
        "don't": "do not",
        "can't": "cannot",
        "won't": "will not",
        "i'm": "i am",
        "you're": "you are",
        "it's": "it is",
        "i've": "i have",
        "we've": "we have",
        "they've": "they have",
        "i'll": "i will",
        "you'll": "you will",
        "they'll": "they will",
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "couldn't": "could not",
        "shouldn't": "should not",
        "wouldn't": "would not",
        "doesn't": "does not",
        "didn't": "did not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "that'll": "that will",
        "??":'?'
    }

    # Escaping all keys before joining them into regex
    pattern = re.compile(r'\b(' + '|'.join(re.escape(k) for k in contractions_dict.keys()) + r')\b', flags=re.IGNORECASE)
    return pattern.sub(lambda x: contractions_dict[x.group().lower()], text)



def preprocess_query(query):
    query = expand_contractions(query)
    query = query.lower()
    return query

# Example usage
multi_intent_dataset = generate_multi_intent_data(intents, total_samples=100, max_combination_size=3)

save_to_csv(multi_intent_dataset, os.path.join(PROCESSED_DIR,f"synthetic_intents.csv"))
# Preview
# for i in range(5):
#     print(multi_intent_dataset[i])

# if __name__ == "__main__":
#     save_to_csv(multi_intent_dataset, os.path.join(PROCESSED_DIR,f"synthetic_intents.csv"))
