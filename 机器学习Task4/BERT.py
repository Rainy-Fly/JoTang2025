import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    TrainerCallback)
from sklearn.metrics import accuracy_score, f1_score

#超参数
max_length=128                 
batch=32                    
lr=3e-5               
epochs=3                        
weight_decay=0.01                
warmup_ratio=0.1                 
seed=42                                       
patience=2           

#种子
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSESEED"]=str(seed)

#预处理
RawData=load_dataset("glue","sst2")  

def clear(example):
    return example["sentence"] is not None and len(example["sentence"].strip()) > 0

RawData=RawData.filter(clear)

tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")

def Tokenize(examples):
    return tokenizer(examples["sentence"],truncation=True,max_length=max_length,padding=False)

tokenized=RawData.map(Tokenize, batched=True).remove_columns(["sentence", "idx"])

#Trainer中动态padding的函数
data_collator=DataCollatorWithPadding(tokenizer=tokenizer,padding=True,return_tensors="pt")

#评估
def evaluate(eval_pred):
    logits,labels=eval_pred
    preds=logits.argmax(axis=-1)
    acc=accuracy_score(labels,preds)
    f1=f1_score(labels,preds,average="binary")
    return {"accuracy":acc,"f1":f1}

#打印日志
class Log(TrainerCallback):
    def __init__(self,interval=50):
        self.print_every=interval
    def on_step_end(self,args,state,control,**kwargs):
        if state.global_step%self.print_every==0 and state.global_step>0:
            logs=state.log_history[-1] if state.log_history else {}
            loss=logs.get("loss", 0.0)
            lr=logs.get("learning_rate", 0.0)
            print(f"[Step {state.global_step}]: loss={loss:.4f},lr={lr:.2e}")

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2, )

Args=TrainingArguments(
    output_dir="./temp/sst2",
    eval_strategy="epoch",  
    save_strategy="epoch", 
    learning_rate=lr,
    per_device_train_batch_size=batch,
    per_device_eval_batch_size=batch,
    num_train_epochs=epochs,
    weight_decay=weight_decay,
    warmup_ratio=warmup_ratio,
    load_best_model_at_end=True,  
    metric_for_best_model="accuracy",
    seed=seed)

trainer = Trainer(
    model=model,
    args=Args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer, 
    data_collator=data_collator,
    compute_metrics=evaluate,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=patience),Log(interval=50)])

#开始训练
trainer.train()
trainer.save_model("./best/sst2") 
eval_res=trainer.evaluate()
print("\n最终指标:",eval_res)
