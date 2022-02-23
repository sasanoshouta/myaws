# 使うモジュール、関数を最初に定義
# train.pyの関数定義部
from pathlib import Path
import re
import math
import time
import copy
from tqdm import tqdm
import pandas as pd
import tarfile
import neologdn
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Model
import settings
import gc
import numpy as np
import boto3
from transformers import pipeline
from io import StringIO

BUCKET = 'sagemaker-us-east-2-523358537305'
pref = 'test_items/data/'
FILE_NAME = 'oran_train_to_T5_hand_oran_1_7.csv'

# 使用するデータ（csvファイル）を読み込む関数
def read():
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=pref+FILE_NAME)
    body = obj['Body']
    csv_string = body.read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_string), index_col=0).rename(columns={'input': 'title', '確認事項': 'body'})
    no_data = [i for i in range(len(df)) if df.loc[i, 'title'] is np.nan]
    new_df = df.drop(df.index[no_data]).reset_index(drop=True)
    return new_df

# 読み込んだdataframeをtrain_iterとvalid_iterに分割する関数
def valid_train_test(df, tokenizer):
    random_state = 94
    X_train, X_test, y_train, y_test = train_test_split(
        df['title'], df['body'], test_size=0.2, random_state=random_state, shuffle=True
    )

    train_data = [(src, tgt) for src, tgt in zip(X_train, y_train)]
    valid_data = [(src, tgt) for src, tgt in zip(X_test, y_test)]

    train_iter, valid_iter = convert_batch_data(train_data, valid_data, tokenizer)
    return train_iter, valid_iter

def convert_batch_data(train_data, valid_data, tokenizer):

    def generate_batch(data):

        batch_src, batch_tgt = [], []
        for src, tgt in data:
            batch_src.append(src)
            batch_tgt.append(tgt)

        batch_src = tokenizer(
            batch_src, max_length=settings.max_length_src, truncation=True, padding="max_length", return_tensors="pt"
        )
        batch_tgt = tokenizer(
            batch_tgt, max_length=settings.max_length_target, truncation=True, padding="max_length", return_tensors="pt"
        )

        return batch_src, batch_tgt
    
    train_iter = DataLoader(train_data, batch_size=settings.batch_size_train, shuffle=True, collate_fn=generate_batch)
    valid_iter = DataLoader(valid_data, batch_size=settings.batch_size_valid, shuffle=True, collate_fn=generate_batch)

    return train_iter, valid_iter

class T5FineTuner(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.model = T5ForConditionalGeneration.from_pretrained(settings.MODEL_NAME)

    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None,
        decoder_attention_mask=None, labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )

def train(model, data, optimizer, PAD_IDX):
    
    model.train()
    
    loop = 1
    losses = 0
    pbar = tqdm(data)
    for src, tgt in pbar:
                
        optimizer.zero_grad()
        
        labels = tgt['input_ids'].to(settings.device)
        labels[labels[:, :] == PAD_IDX] = -100

        outputs = model(
            input_ids=src['input_ids'].to(settings.device),
            attention_mask=src['attention_mask'].to(settings.device),
            decoder_attention_mask=tgt['attention_mask'].to(settings.device),
            labels=labels
        )
        loss = outputs['loss']

        loss.backward()
        optimizer.step()
        losses += loss.item()
        
#         pbar.set_description("src: %d " % src)
        pbar.set_postfix(loss=losses / loop)
        loop += 1
        wandb.log({"train_loss": losses / loop})
    return losses / len(data)

def evaluate(model, data, PAD_IDX):
    
    model.eval()
    losses = 0
    with torch.no_grad():
        for src, tgt in data:

            labels = tgt['input_ids'].to(settings.device)
            labels[labels[:, :] == PAD_IDX] = -100

            outputs = model(
                input_ids=src['input_ids'].to(settings.device),
                attention_mask=src['attention_mask'].to(settings.device),
                decoder_attention_mask=tgt['attention_mask'].to(settings.device),
                labels=labels
            )
            loss = outputs['loss']
            losses += loss.item()
            
            wandb.log({"valid_loss": losses})
    return losses / len(data)

def generate_text_from_model(title, trained_model, tokenizer, num_return_sequences=1):

    trained_model.eval()
    
    title = preprocess_text(title)
    batch = tokenizer(
        [title], max_length=settings.max_length_src, truncation=True, padding="longest", return_tensors="pt"
    )

    # 生成処理を行う
    outputs = trained_model.generate(
        input_ids=batch['input_ids'].to(settings.device),
        attention_mask=batch['attention_mask'].to(settings.device),
        max_length=settings.max_length_target,
        repetition_penalty=8.0,   # 同じ文の繰り返し（モード崩壊）へのペナルティ
        # temperature=1.0,  # 生成にランダム性を入れる温度パラメータ
        num_beams=25,  # ビームサーチの探索幅
        diversity_penalty=1.0,  # 生成結果の多様性を生み出すためのペナルティパラメータ
        num_beam_groups=25,  # ビームサーチのグループ
        num_return_sequences=num_return_sequences,  # 生成する文の数
    )

    generated_texts = [
        tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) for ids in outputs
    ]

    return generated_texts

def preprocess_text(text):
    text = re.sub(r'[\r\t\n\u3000]', '', text)
    text = neologdn.normalize(text)
    text = text.lower()
    text = text.strip()
    return text

# BARTの事前学習済みモデルを使って抜き出した文章を指定単語長に要約する
def txt_summarization_by_bart(df):
    smr_bart = pipeline(task=settings.task_name, model=settings.sum_model_from_bart)
    records = len(df)
    for record in range(records):
        src_text = df.loc[record, 'title']
        smbart = smr_bart(src_text, max_length=settings.max_length_src, truncation=True)
        df.loc[record, 'title'] = smbart[0]["summary_text"]
    
    return df

# コサイン類似度計算
def vectorize(text: str):
#     encoding = tokenizer(text, padding = 'longest', return_tensors='pt')
#     outputs = model(**encoding)
    tokenizer = T5Tokenizer.from_pretrained(model_dir_path)
    model = T5Model.from_pretrained(model_dir_path)
    input_ids = tokenizer(text, return_tensors="pt").input_ids  # Batch size 1
    decoder_input_ids = tokenizer(text, return_tensors="pt").input_ids  # Batch size 1
    outputs = model(input_ids, decoder_input_ids=decoder_input_ids)

    return outputs.last_hidden_state[0][0]

def cos_sim(v1, v2):
    v1 = v1.detach().numpy()
    v2 = v2.detach().numpy()
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# train.pyのmain部
if __name__ == "__main__":

    # load datasets
    df = read()
    # BARTで、抜き出してきた文章の要約を行う関数（T5で文章生成による学習を行う際に、重要と思われる情報に短くまとめる為）
    new_df = txt_summarization_by_bart(df)
    # T5Tokenizer定義
    tokenizer = T5Tokenizer.from_pretrained(settings.MODEL_NAME, is_fast=True)
    # train_iter, valid_iterに分割
    train_iter, valid_iter = valid_train_test(df, tokenizer)
    
    model = T5FineTuner()
    model = model.to(settings.device)

    optimizer = optim.Adam(model.parameters())

    PAD_IDX = tokenizer.pad_token_id
    best_loss = float('Inf')
    best_model = None
    counter = 1
    gc.collect()
    
    for loop in range(1, settings.epochs + 1):
        start_time = time.time()
        loss_train = train(model=model, data=train_iter, optimizer=optimizer, PAD_IDX=PAD_IDX)
        elapsed_time = time.time() - start_time
        loss_valid = evaluate(model=model, data=valid_iter, PAD_IDX=PAD_IDX)

        print('[{}/{}] train loss: {:.4f}, valid loss: {:.4f} [{}{:.0f}s] counter: {} {}'.format(
            loop, settings.epochs, loss_train, loss_valid,
            str(int(math.floor(elapsed_time / 60))) + 'm' if math.floor(elapsed_time / 60) > 0 else '',
            elapsed_time % 60,
            counter,
            '**' if best_loss > loss_valid else ''
        ))

        if best_loss > loss_valid:
            best_loss = loss_valid
            best_model = copy.deepcopy(model)
            counter = 1
        else:
            if counter > settings.patience:
                break
            counter += 1
        gc.collect()
