import numpy as np
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from torch.utils.data import Dataset
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

train = pd.read_csv('/home/vboxuser/mlops4/datasets/train.csv')


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


model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
  output_dir="/home/vboxuser/mlops4/model",
  per_device_train_batch_size=parameters['batch_size'],
  evaluation_strategy="epoch",
  logging_strategy = "epoch",
  save_strategy =  "epoch",
  num_train_epochs=parameters['num_train_epochs'],
  learning_rate=parameters['learning_rate'],
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  load_best_model_at_end=True,
)


def collate_fn(batch):
    return {
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=train_set,
    eval_dataset=val_set
)


train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_state()
f1 = trainer.evaluate(val_set)['eval_f1']
Logger.current_logger().report_scalar(title='F1', series='F1', value=f1, iteration=1)
task.upload_artifact(name='F1', artifact_object={'F1': f1})