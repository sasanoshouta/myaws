  
# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import io
import json
import os
import pickle
from tensorflow.keras import models
import signal
import sys
import traceback
import numpy as np

import flask
import pandas as pd

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")
MODEL_PATH = "../ml/model/model_1.h5"

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.


class ScoringService:
    
    def __init__(self):
        self.model = models.load_model(MODEL_PATH)  # Where we keep the model when it's loaded

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
        if str(type(data)) != "<class 'numpy.ndarray'>":
            data = np.array(data)
            data = data.reshape(data.shape[0], data.shape[1], 1)
        else:
            pass
        result = self.model.predict(data)
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
    s = io.StringIO(data)
    data = pd.read_csv(s)

    # Convert from CSV to pandas
    # if flask.request.content_type == "text/csv":
    #     data = flask.request.data.decode("utf-8")
    #     s = io.StringIO(data)
    #     data = pd.read_csv(s, header=None)
    # else:
    #     return flask.Response(
    #         response="This predictor only supports CSV data", status=415, mimetype="text/plain"
    #     )

    print("Invoked with {} records".format(data.shape[0]))

    # Do the prediction
    scoringservice = ScoringService()
    predictions = scoringservice.predict(data)
    # Convert from numpy back to CSV
    out = io.StringIO()
    pd.DataFrame(predictions).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return flask.Response(response=result, status=200, mimetype="text/csv")