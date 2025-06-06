import os
import sys
import pickle

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.paths_config import PROCESSED_DIR, MODEL_DIR
from src.custom_model import CustomDataset, BertForMultiLabel
import torch
from transformers import BertTokenizer

class ScoringService(object):
    model = None

    @classmethod
    def get_model(cls):
        with open(os.path.join(MODEL_DIR,'labels.pkl' ), 'rb') as inp:
            intent_labels = pickle.load(inp)
            cls.mlb = pickle.load(inp)
        cls.model = BertForMultiLabel(num_labels=len(intent_labels))
        cls.model.load_state_dict(torch.load(os.path.join(MODEL_DIR,"bert_multilabel_model.pth")))
        cls.model.eval() 
        cls.tokenizer = BertTokenizer.from_pretrained(os.path.join(MODEL_DIR,"bert_tokenizer/"))
        return cls.model, cls.tokenizer, cls.mlb
    

    @classmethod
    def predict(cls,text):
        model, tokenizer, mlb =  cls.get_model()
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            probs = torch.sigmoid(outputs)
            preds = outputs[0].numpy()
            print(preds,probs)
            intents = [mlb.classes_[i] for i, p in enumerate(preds) if p > 0.8]
        return intents
    

if __name__ == '__main__':
    predictions = ScoringService.predict('How can I get a refund for my order #56789?')
    print(predictions)
    # A zero exit code causes the job to be marked a Succeeded.
    sys.exit(0)