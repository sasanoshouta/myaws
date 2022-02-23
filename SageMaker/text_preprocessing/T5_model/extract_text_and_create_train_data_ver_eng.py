import boto3
import gc
import pandas as pd
import re
import nltk
if nltk.download('stopwords') == True:
    pass
else:
    nltk.download('stopwords')

from nltk.corpus import stopwords

# ストップワードの取得
stop_words = stopwords.words('english')
# データ読み込み
new_sample = pd.read_excel("./sample.xlsx", engine='openpyxl', header=1).reset_index(drop=True)

# S3から画像化したpdfイメージの名前取得をしてAWS TextractにかけてOCRしてページ毎にテキストを補完する
# Amazon Textract client
textract = boto3.client('textract', region_name="us-east-2")
s3 = boto3.client('s3')
Bucket = 'sagemaker-us-east-2-523358537305'
OBJ_filter = 'pdf/image/suiikei'
image_name_list = list()

response = s3.list_objects_v2(Bucket=Bucket)
for object in response['Contents']:
    if OBJ_filter not in object['Key']:
        pass
    else:
        image_name_list.append(object['Key'])
        
image_name_list = sorted(image_name_list, key=lambda s: int(re.search(r'\d+', s).group()))
TABLE = 'Table'
text_dict = dict()

for i, page in enumerate(image_name_list):
    text_tmp = list()
    body = s3.get_object(Bucket=Bucket,Key=page)['Body'].read()
    response = textract.detect_document_text(Document={'Bytes': body})
    for item in response['Blocks']:
        if 'Text' in item.keys():
            text_tmp.append(item['Text'])
        else:
            pass
        
    if len(text_tmp) == 0:
        pass
    else:
        text_dict[i] = ' '.join(text_tmp)
        
# テキスト抽出後の処理
DELETE_SENTENCE = ['O-RAN A L L I A N C E ORAN-WG4.CUS.0-v02.0', 
                   'ORAN-WG4.CUS.0-v02.00 I-RAN A L L I A N C ', 
                   'ORAN-WG4.CUS.0-v02.00 -RAN A N E', 
                   'ORAN-WG4.CUS.0-v02.00 RAN A L I A N C E', 
                   'ORAN-WG4.CUS.0-v02.00 D-AR A L L I A N C E', 
                   'ORAN-WG4.CUS.0-v02.00 RAN A L L I A N C E ',
                   'ORAN-WG4.CUS.0-v02.00 RAN L I A N C E', 
                   'ORAN-WG4.CUS.0-v02.00 A L L I A N C E', 
                   'ORAN-WG4.CUS.0-v02.00 -RAN L L I A N C E', 
                   'ORAN-WG4.CUS.0-v02.00 -RAN A L I A N C E', 
                   'ORAN-WG4.CUS.0-v02.00 RAN L L I A N C E', 
                   'ORAN-WG4.CUS.0-v02.00 -RAN L I A N C E',
                   'ORAN-WG4.CUS.0-v02.00 -RAN A L L I A N C E', 
                   'ORAN-WG4.CUS.0-v02.00 -RAN A']
# 抽出文章を補正する関数
def fix(text_dict: dict):
    new_dict = dict()
    for page in range(len(text_dict)):
        result = text_dict[page]
        tmp = re.findall('Copyright © 2019 by the O-RAN Alliance. Your use is subject to the terms of the O-RAN Adopter License Agreement in Annex ZZZ [0-9]+', text_dict[page])
        tmp = list(set(tmp))
        result = result.replace(tmp[0], "")
        for sent in DELETE_SENTENCE:
            if sent in text_dict[page]:
                result = result.replace(sent, "")
                
        new_dict[page] = result
    return new_dict

new_dict = fix(text_dict)

# AWS TextractでOCRした結果から作成する学習用文章データ
for i in range(len(new_sample)):
    check = sample['input'][i]
    tmp_text = list()
    for c_text in check:
        for page in new_dict.keys():
            if c_text in new_dict[page]:
                tmp_text.append(new_dict[page])
            else:
                pass
    new_sample.loc[i, 'input'] = tmp_text
    
train_data = new_sample.drop(['Reference', '項番'], axis=1)
train_data.to_csv('train.csv')
