
import re
import os
import sys
import pandas as pd
import numpy as np
from ast import literal_eval
import pickle
from sklearn.metrics import hamming_loss, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.class_weight import compute_class_weight

from custom_model import CustomDataset, BertForMultiLabel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.paths_config import PROCESSED_DIR, MODEL_DIR
from config.paths_config import PROCESSED_DIR
    


    
def train():
    df = pd.read_csv(os.path.join(PROCESSED_DIR,'synthetic_intents.csv'))
    df['intents'] = df['intents'].apply(literal_eval)

    mlb = MultiLabelBinarizer()
    df['label_vector'] = mlb.fit_transform(df['intents']).tolist()
    intent_labels = mlb.classes_
    print(intent_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Compute weights using sklearn utility or normalized inverse frequency

    # Flatten your label list
    import numpy as np
    labels = np.array(df['label_vector'].tolist())

    # Sum over axis to get counts per class
    label_counts = labels.sum(axis=0)

    # Normalize inverse frequency
    class_weights = len(labels) / (label_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_weights)

    # Convert to tensor
    pos_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(df['text'], df['label_vector'], test_size=0.3, random_state=42)
    val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def encode(texts):
        return tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")

    train_encodings = encode(train_texts)
    val_encodings = encode(val_texts)
    test_encodings = encode(test_texts)

    train_dataset = CustomDataset(train_encodings, train_labels.tolist())
    val_dataset = CustomDataset(val_encodings, val_labels.tolist())
    test_dataset = CustomDataset(test_encodings, test_labels.tolist())

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    model = BertForMultiLabel(num_labels=len(intent_labels))
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    #------------------------------------
    #weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    model.train()
    for epoch in range(20):
        print(f"Epoch {epoch + 1}")
        total_train_loss = 0

        for batch in train_loader:
            model.train()
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

    # Validation phase
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    def evaluate(model, dataloader, device):#device
        model.eval()
        preds, true_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                #logits = outputs.logits
                sigmoid_logits = torch.sigmoid(outputs)
                preds.extend(sigmoid_logits.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        preds = np.array(preds)
        true_labels = np.array(true_labels)
        binarized_preds = (preds >= 0.5).astype(int)

        metrics = {
            'Hamming Loss': hamming_loss(true_labels, binarized_preds),
            'Precision': precision_score(true_labels, binarized_preds, average='macro', zero_division=0),
            'Recall': recall_score(true_labels, binarized_preds, average='macro', zero_division=0),
            'F1 Score': f1_score(true_labels, binarized_preds, average='macro', zero_division=0),
            'Subset Accuracy': accuracy_score(true_labels, binarized_preds)
        }
        return metrics
    test_metrics = evaluate(model, test_loader, device)#device

    print("Test Metrics:", test_metrics)

    def predict(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.sigmoid(outputs)
    #         print(probs)
    #         print(outputs)
            preds = outputs[0].numpy()
            #print(preds)
            intents = [mlb.classes_[i] for i, p in enumerate(probs[0]) if p > 0.8]
        return intents

    print(predict("I’d like to request a refund for my recent order and also get more details about your latest laptop models."))
    print(predict('would like to get details about product i received is different from the one that i placed shipment, need help with repayment. i have been waiting for the order long time do you have this product in stock?'))
    print(predict('How can I get a refund for my order #56789?'))


    # Save model weights and tokenizer
    model_save_path = "bert_multilabel_model.pth"
    torch.save(model.state_dict(), os.path.join(MODEL_DIR,model_save_path))

    # Optional: Save tokenizer
    tokenizer.save_pretrained(os.path.join(MODEL_DIR,"bert_tokenizer/"))

    with open(os.path.join(MODEL_DIR, 'labels.pkl'), 'wb') as out:
        pickle.dump(intent_labels, out)
        pickle.dump(mlb,out) 


if __name__ == '__main__':
    train()

    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)
