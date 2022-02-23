# 画像化したpdfイメージからAWS Textractで文字抽出
import boto3
s3 = boto3.client('s3')
Bucket = 'buclet_name'
OBJ_filter = 'obj_filter'
image_name_list = list()

response = s3.list_objects_v2(Bucket=Bucket)
for object in response['Contents']:
    if OBJ_filter not in object['Key']:
        pass
    else:
        image_name_list.append(object['Key'])
        
        
TABLE = 'Table'
table_dict = dict()

for page in image_name_list:
    text_tmp = list()
    body = s3.get_object(Bucket=Bucket,Key=page)['Body'].read()
    response = textract.detect_document_text(Document={'Bytes': body})
    for item in response['Blocks']:
        if (item['BlockType'] == 'LINE') and (item['Text'].startswith(TABLE) == True):
            text_tmp.append(item['Text'])
    if len(text_tmp) == 0:
        pass
    else:
        table_dict[page] = text_tmp
        
response = textract.detect_document_text(
    Document={
        'Bytes': body
    }
)

# Print detected text
for item in response["Blocks"]:
    if item["BlockType"] == "LINE":
        print ('\033[94m' +  item["Text"] + '\033[0m')
