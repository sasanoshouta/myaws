# ファイルをフォルダごとアップロードするスクリプト

import boto3
import os
import sys

client = boto3.client('s3')
bucket = 'mybucket'

def upload_files(basedir):
    parent_dir = os.path.dirname(os.path.realpath(basedir))
    for (path, dirs, files) in os.walk(basedir):
        for fn in files:
            if fn.startswith('.'):
                continue
            abspath = os.path.join(path, fn)
            yield (
                abspath,
                os.path.relpath(abspath, parent_dir).split(''.join([basedir, '/']))[1]
            )
            
dirname = '保存したいローカルディレクトリ名'
pref = 'S3で切りたいprefix'
for abspath, relpath in upload_files(dirname):
    client.upload_file(abspath, bucket, pref+abspath)
