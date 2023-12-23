import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import evaluate
from clearml import Task, Logger

task = Task.init(project_name='disaster_tweets', task_name='module use HP')
logger = task.get_logger()    

parameters = {
    'train_test_split' : 0.3,
    'batch_size' : 8,
    'num_train_epochs' : 1,
    'learning_rate' : 1e-5,
}

parameters = task.connect(parameters)

train = pd.read_csv('/content/train.csv')


def cleaner(text):
    pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
    text = re.sub(pat,'',text)
    text = re.sub(r'@\S+',' ',text)
    text = re.sub(r'[ÛªÛÒÏâÂÓåÈ]', '', text)
    text = re.sub(r'\n','',text)
    text = re.sub(r'\t','',text)
    text = re.sub("[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub("http", "", text)
    text = re.sub(r'\b\w\b', "", text)
    text = re.sub('RT', "", text)
    text = re.sub(r'UTC', "", text)
    text = re.sub(r'km', "", text)
    return(text)


train['text'] = train['text'].apply(cleaner)
train['text'].replace('', np.nan, inplace=True)
train.dropna(subset=['text'], inplace=True)
train = train.applymap(lambda x: x.lower() if isinstance(x, str) else x)
train.drop(columns=['id','keyword','location'],inplace=True)
train.rename(columns= {'target': 'label'}, inplace=True)


tweets = train.text.values
labels = train.label.values


model_id = "prajjwal1/bert-mini"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

X_train, X_valid, y_train, y_valid = train_test_split(tweets, labels, test_size=parameters['train_test_split'], random_state=42)


train_encodings = []
val_encodings = []

for text in X_train:
    train_encodings.append((tokenizer.encode_plus(text, add_special_tokens = True, max_length = 40, pad_to_max_length=True, truncation=True)))
for text in X_valid:
    val_encodings.append((tokenizer.encode_plus(text, add_special_tokens = True, max_length = 40, pad_to_max_length=True, truncation=True)))


def process_encodings(encodings):
    input_ids=[]
    attention_mask=[]
    for enc in encodings:
        input_ids.append(enc['input_ids'])
        attention_mask.append(enc['attention_mask'])
    return {'input_ids':input_ids, 'attention_mask':attention_mask}


train_ready = process_encodings(train_encodings)
val_ready = process_encodings(val_encodings)


class MyCustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        assert len(self.encodings['input_ids']) == len(self.encodings['attention_mask']) ==  len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    

train_set = MyCustomDataset(train_ready,y_train)
val_set = MyCustomDataset(val_ready,y_valid)

train_dataloader = DataLoader(train_set, shuffle = True, batch_size=parameters['batch_size'])
val_dataloader = DataLoader(val_set, shuffle = False, batch_size=parameters['batch_size'])

def train(model, train_dataloader, val_dataloader, optimizer, epochs=10, device='cpu', threshold = 0.5):
    global best_f1
    print(f'Training for {epochs} epochs on {device}')
    for epoch in range(1,epochs+1):
        print(f"Epoch {epoch}/{epochs}")

        model.train()
        model.zero_grad() 
        train_loss = 0
        train_bar = tqdm(train_dataloader, total=len(train_dataloader))
        for step, batch in enumerate(train_bar):
            tokens_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            loss, logits = model(input_ids = tokens_ids, attention_mask=attention_mask, labels = labels)[:2]
            train_bar.set_description("epoch {} loss {}".format(epoch,loss))
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if val_dataloader is not None:
            logits = []
            y_trues = []
            y_preds = []
            model.eval()

            valid_loss = 0
            val_bar = tqdm(val_dataloader,total=len(val_dataloader))
            for step, batch in enumerate(val_bar):
                with torch.no_grad():
                    tokens_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    loss, logits = model(input_ids = tokens_ids, attention_mask=attention_mask, labels = labels)[:2]
                    valid_loss += loss.item()

                    y_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
                    y_trues.append(labels.cpu().numpy())


            y_trues = np.concatenate(y_trues,0)
            y_preds = np.concatenate(y_preds,0)
            train_loss = train_loss/len(train_dataloader.dataset)
            valid_loss = valid_loss/len(val_dataloader.dataset)

            precision = precision_score(y_trues, y_preds)
            recall = recall_score(y_trues, y_preds)
            f1 = f1_score(y_trues, y_preds)
            if f1>best_f1:
                best_f1=f1

best_f1 = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = parameters['learning_rate']
epochs = parameters['num_train_epochs']
model = model.to(device)
optimizer = AdamW(params=model.parameters(), lr=lr, eps = 1e-8)

train(model = model,
      train_dataloader = train_dataloader,
      val_dataloader = val_dataloader,
      optimizer = optimizer,
      epochs = epochs,
      device = device)

Logger.current_logger().report_scalar(title='F1', series='F1', value=best_f1, iteration=1)
task.upload_artifact(name='F1', artifact_object={'F1': best_f1})
