import pandas as pd
import numpy as np

from datasets import Dataset

df = pd.read_table('./sms_spam_collection/SMSSpamCollection', header=None, encoding="UTF-8")
df.columns = ['label', 'text']

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

ds = Dataset.from_pandas(df)

from transformers import AutoModelForSequenceClassification,AutoTokenizer

model_nm = "bert-base-uncased"
tokz = AutoTokenizer.from_pretrained(model_nm, use_auth_token=False, trust_remote_code=True, local_files_only=False)

def tok_func(x): return tokz(x["text"])

tok_ds = ds.map(tok_func, batched=True)

tok_ds = tok_ds.rename_columns({'label':'labels'})

dds = tok_ds.train_test_split(0.25, seed=42)

train_dataset = dds['train']
test_dataset = dds['test']

train_val_split = train_dataset.train_test_split(0.25, seed=42) 

train_dataset = train_val_split['train']
val_dataset = train_val_split['test']

from transformers import TrainingArguments,Trainer

bs = 128
epochs = 10
lr = 0.0001

args = TrainingArguments(
    'outputs',
    learning_rate=lr,
    warmup_steps=500, 
    fp16=True,
    per_device_train_batch_size=bs,
    per_device_eval_batch_size=bs*2,
    num_train_epochs=epochs,
    weight_decay=0.01
)

from scipy.stats import pearsonr

def corr(preds, labels):
    preds = np.argmax(preds, axis=1)
    corr_coef, _ = pearsonr(preds, labels) 
    return corr_coef

def corr_d(eval_pred):
    preds, labels = eval_pred
    return {'pearson': float(corr(preds, labels))} 

model = AutoModelForSequenceClassification.from_pretrained(model_nm, num_labels=2)
trainer = Trainer(model, args, train_dataset=train_dataset, eval_dataset=val_dataset,
                  tokenizer=tokz, compute_metrics=corr_d)

trainer.train()





