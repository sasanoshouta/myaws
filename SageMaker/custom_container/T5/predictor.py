  
# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import io
import json
import os
import pickle
import signal
import sys
import traceback
import numpy as np
import flask
import pandas as pd

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

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")
MODEL_PATH = "../ml/model/"

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
def preprocess_text(text):
    text = re.sub(r'[\r\t\n\u3000]', '', text)
    text = neologdn.normalize(text)
    text = text.lower()
    text = text.strip()
    return text
    
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
    result_dict = dict()
    for i in range(len(generated_texts)):
        result_dict[i] = generated_texts[i]
    result = json.dumps(result_dict)

    return result

class t5_model:
    
    def __init__(self):
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
        self.trained_model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH) # Where we keep the model when it's loaded

    # @classmethod
    # def get_model(cls):
    #     """Get the model object for this instance, loading it if it's not already loaded."""
    #     if cls.model == None:
    #         with open(os.path.join(model_path, "model_1.h5"), "rb") as inp:
    #             cls.model = models.load_model(inp)
    #     return cls.model

    # @classmethod
    # def predict(cls, input):
    #     """For the input, do the predictions and return them.
    #     Args:
    #         input (a pandas dataframe): The data on which to do the predictions. There will be
    #             one prediction per row in the dataframe"""
    #     clf = cls.get_model()
    #     return clf.predict(input)
    def predict(self, data):
        
        result = generate_text_from_model(title=data, trained_model=self.trained_model, tokenizer=self.tokenizer, num_return_sequences=10)
        return result


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here

    # status = 200 if health else 404
    status = 200
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = flask.request.data.decode("utf-8")
    # s = io.StringIO(data)
    # data = pd.read_csv(s)

    # Convert from CSV to pandas
    # if flask.request.content_type == "text/csv":
    #     data = flask.request.data.decode("utf-8")
    #     s = io.StringIO(data)
    #     data = pd.read_csv(s, header=None)
    # else:
    #     return flask.Response(
    #         response="This predictor only supports CSV data", status=415, mimetype="text/plain"
    #     )

    # print("Invoked with {} records".format(data.shape[0]))
    print("Invoked with {}".format(data))

    # Do the prediction
    model = t5_model()
    result = model.predict(data)
    # Convert from numpy back to CSV
    # out = io.StringIO()
    # pd.DataFrame(predictions).to_csv(out, header=False, index=False)
    # result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype="text/csv")