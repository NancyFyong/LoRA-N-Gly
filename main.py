import argparse
from scipy.special import softmax
from transformers import (AutoTokenizer,Trainer
,TrainingArguments
,DataCollatorWithPadding,
EarlyStoppingCallback)
from peft import LoraConfig, get_peft_model,PeftModel
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
import math
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from accelerate import Accelerator
import numpy as np
import logging
from model.esm_model import EsmModelClassification
from sklearn.metrics import f1_score, accuracy_score, average_precision_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,confusion_matrix
import numpy as np
parser = argparse.ArgumentParser(description='Train or evaluate LoRA-N-Gly.')
parser.add_argument('--model_name', default='facebook/esm2_t36_3B_UR50D', type=str, help='Base ESM-2 model name or path')
parser.add_argument('--stage', default='test', type=str, help='train or test')
parser.add_argument('--train_dataset', default='./data/N-GlycositeAltas/train.csv', type=str, help='Path to train dataset')
parser.add_argument('--valid_dataset',default='./data/N-GlycositeAltas/valid.csv',type=str,help='')
parser.add_argument('--test_dataset', default='./data/N-GlycositeAltas/test.csv', type=str, help='Path to validation dataset')
parser.add_argument('--peft_model_path', default='./checkpoints/N-linked/ESM-3B/checkpoint', type=str, help='Path to PEFT model checkpoint')
parser.add_argument('--save_dir',default='./checkpoint')
args = parser.parse_args()
# 设置日志级别
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 读取数据集

logging.info(f"Loading train dataset from {args.train_dataset}")
train_df = pd.read_csv(args.train_dataset)
valid_df = pd.read_csv(args.valid_dataset)
test_df = pd.read_csv(args.test_dataset)

logging.info(f"Train dataset size: {len(train_df)}")
logging.info(f"Validation dataset size: {len(valid_df)}")
logging.info(f"Test dataset size: {len(test_df)}")

logging.info(f"Model name: {args.model_name}")
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

model = EsmModelClassification.from_pretrained(args.model_name,num_labels=2,device_map='auto')

logging.info("Model loaded successfully")

logging.info("Setting up data collator")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from datasets import Dataset

data_train = Dataset.from_pandas(train_df)
data_valid = Dataset.from_pandas(valid_df)
data_test = Dataset.from_pandas(test_df)

def preprocess(example):
    tokenized_example = tokenizer(example["sequence"])
    tokenized_example['labels'] = example['label']
    # pos is the target candidate residue position used by the classification head.
    tokenized_example['pos'] = example['pos']
    
    return tokenized_example


data_train = data_train.map(preprocess, remove_columns=data_train.column_names, batched=True)
data_valid = data_valid.map(preprocess, remove_columns=data_valid.column_names, batched=True)
data_test = data_test.map(preprocess, remove_columns=data_test.column_names, batched=True)



def compute_metrics(eval_preds):
   
    logits, labels = eval_preds
    probability = softmax(logits, axis=1)[:,-1]
    predictions = np.argmax(logits, axis=-1)
    unique_labels = np.unique(labels)
    f1 = f1_score(y_true=labels, y_pred=predictions, zero_division=0)
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    mcc = matthews_corrcoef(y_true=labels, y_pred=predictions)
    auc_roc = roc_auc_score(y_true=labels, y_score=probability) if len(unique_labels) == 2 else float('nan')
    auc_pr = average_precision_score(y_true=labels, y_score=probability) if len(unique_labels) == 2 else float('nan')
    precision = precision_score(y_true=labels, y_pred=predictions, zero_division=0)
    recall = recall_score(y_true=labels, y_pred=predictions, zero_division=0)
    TN, FP, FN, TP = confusion_matrix(y_true=labels, y_pred=predictions, labels=[0, 1]).ravel()
    specificity = TN / (TN + FP) if (TN + FP) else 0.0
    metrics = {'f1': f1,
               'accuracy': accuracy,
               'mcc': mcc,
               'auc_roc': auc_roc,
               'auc_pr': auc_pr,
               'precision': precision,
               'recall': recall,
               'SN': recall,
               'specificity': specificity,
               'SP': specificity,
               'TP': int(TP),
               'FP': int(FP),
               'TN': int(TN),
               'FN': int(FN)}
    return metrics


training_args = TrainingArguments(
    output_dir=args.save_dir,
    report_to='tensorboard',
    overwrite_output_dir=True,
    warmup_ratio=0.2,
    lr_scheduler_type='cosine',
    learning_rate=3e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    metric_for_best_model='eval_mcc',
    num_train_epochs=50,
    evaluation_strategy='steps',
    save_strategy='steps',
    logging_steps=500,
    eval_steps=1000,
    save_steps=1000,
    load_best_model_at_end=True,
    weight_decay=0.1,
    save_total_limit=10
)


def get_trainer(model):
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=data_train,
        eval_dataset=data_valid,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
	    callbacks = [EarlyStoppingCallback(early_stopping_patience=10)],
    )


peft_config = LoraConfig(
    task_type="SEQ_CLS",
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias='all',
    target_modules=['query', 'value', 'out_proj'])



if args.stage == 'train':
    print('PEFT Model')
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    peft_lora_finetuning_trainer = get_trainer(peft_model)
    peft_lora_finetuning_trainer.train()
else:
    peft_model = PeftModel.from_pretrained(model, args.peft_model_path)
    peft_lora_finetuning_trainer = get_trainer(peft_model)
    predictions = peft_lora_finetuning_trainer.predict(data_test)
    logging.info(predictions.metrics)
