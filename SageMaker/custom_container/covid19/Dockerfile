FROM nvidia/cuda:11.0-devel-ubuntu20.04

ENV PROGRAM_DIR=/opt/sasanosan
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir -p $PROGRAM_DIR
WORKDIR $PROGRAM_DIR

RUN apt-get update
RUN apt-get install -y python3 python3-pip tzdata nginx

# 学習/推論の実装に必要なモジュール群をインストールする
RUN pip3 install --no-cache-dir numpy==1.19.5 pandas==1.1.5 torch==1.7.0 torchvision transformers==4.5.1 scikit-learn==0.24.1
RUN pip3 install --no-cache-dir tensorflow

# SageMaker上で使用する AWS SDKのモジュール等をインストールする
RUN pip3 install ipykernel && \
    python3 -m ipykernel install --sys-prefix && \
    pip3 install --quiet --no-cache-dir \
    'boto3>1.0<2.0' \
    'sagemaker>2.0<3.0'
RUN pip3 --no-cache-dir install flask gunicorn

# コンテナ内に学習/推論用スクリプト群を配置する
COPY train.py $PROGRAM_DIR/train
COPY serve.py $PROGRAM_DIR/serve
COPY wsgi.py $PROGRAM_DIR/wsgi.py
COPY predictor.py $PROGRAM_DIR/predictor.py
COPY nginx.conf $PROGRAM_DIR/nginx.conf

# train/serveを実行するためのパスを通す
ENV PATH=$PATH:$PROGRAM_DIR

# train/serveの実行権限を付与する
RUN chmod +x train
RUN chmod +x serve