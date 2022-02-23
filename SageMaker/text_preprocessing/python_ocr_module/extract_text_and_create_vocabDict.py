# PDFから文章を抽出
import gc
import PyPDF2
import re
import nltk
if nltk.download('stopwords') == True:
    pass
else:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import word2vec

# pdfから記号と数字を削除してtextを抽出する関数
def extract_text_from_pdf(file_name):
    text = dict()
    # ORANの補正
    unique_word = ['O RAN', 'O R AN', 'OR AN', 'O R A N', 'O RA N', 'ORA N']
    TRUE_WORD = 'ORAN'
    # copyrightの削除
    COPYRIGHT = 'copyright oran alliance your use subject terms oran adopter license agreement annex zzz oran wg cus v '
    # ストップワードの取得
    stop_words = stopwords.words('english')
    # 記号の削除
    code_regex = re.compile('[\t\s!"#$%&\'\\\\()*+,-./:;；：<=>?@[\\]^_`{|}~○｢｣「」〔〕“”〈〉'\
                '『』【】＆＊（）＄＃＠？！｀＋￥¥％♪…◇→←↓↑｡･ω･｡ﾟ´∀｀ΣДｘ⑥◎©︎♡★☆▽※ゞノ〆εσ＞＜┌┘]')
    # 数字の削除
    num_regex = re.compile('\d+,?\d*')
    with open(file_name, 'rb') as f:
        for i in range(218):
            reader = PyPDF2.PdfFileReader(f)
            page = reader.getPage(i)
            tmp = page.extractText()
#             tmp = " ".join(tmp.split())
            result = code_regex.sub(' ', tmp)
            result = num_regex.sub(' ', result)
            result = " ".join([word for word in result.split() if word not in stop_words])
            for oran in unique_word:
                if oran in result:
                    result = result.replace(oran, TRUE_WORD)
                else:
                    pass
            result = result.lower()
            copyright_removed = result.replace(COPYRIGHT, "")
            text[i] = copyright_removed
    print('finished')
    return text

# LSTM学習の為にピリオドで文章を区切ったデータに成形する関数
def extract_text_from_pdf_for_lstm(file_name):
    text_to_lstm = dict()
    # ORANの補正
    unique_word = ['O RAN', 'O R AN', 'OR AN', 'O R A N', 'O RA N', 'ORA N']
    TRUE_WORD = 'ORAN'
    # copyrightの削除
    drop_copy = ['Copyright O RAN Alliance', ' Your use subject terms O R AN Adopter License Agreement Annex ZZZ ORAN WG ', ' CUS', ' v ']
    COPYRIGHT = 'Copyright ORAN Alliance Your use subject terms ORAN Adopter License Agreement Annex ZZZ ORAN WG CUS v'
    # ストップワードの取得
    stop_words = stopwords.words('english')
    # 記号の削除
    code_regex = re.compile('[\t\s!"#$%&\'\\\\()*+,-./:;；：<=>?@[\\]^_`{|}~○｢｣「」〔〕“”〈〉'\
                '『』【】＆＊（）＄＃＠？！｀＋￥¥％♪…◇→←↓↑｡･ω･｡ﾟ´∀｀ΣДｘ⑥◎©︎♡★☆▽※ゞノ〆εσ＞＜┌┘]')
    # 数字の削除
    num_regex = re.compile('\d+,?\d*')
    with open(file_name, 'rb') as f:
        for i in range(218):
            reader = PyPDF2.PdfFileReader(f)
            page = reader.getPage(i)
            tmp = page.extractText()
            result = tmp.replace('.', 'eos\n')
#             tmp = " ".join(tmp.split())
            result = code_regex.sub(' ', result)
            result = num_regex.sub(' ', result)
            result = " ".join([word for word in result.split() if word not in stop_words])
            result = result.replace('eos', 'eos\n')
            result = result.split('eos\n')
            result = [word for word in result if word not in drop_copy]
            result = [word for word in result if len(word) >= 2]
            for oran in unique_word:
                for j in range(len(result)):
                    result[j] = result[j].replace(COPYRIGHT, "")
                    if oran in result[j]:
                        result[j] = result[j].replace(oran, TRUE_WORD)
                    else:
                        pass
            text_to_lstm[i] = result
    print('finished')
    return text_to_lstm


FILENAME = 'ORAN-WG4.CUS.0-v02.00.pdf'
text = extract_text_from_pdf(FILENAME)
text_to_lstm = extract_text_from_pdf_for_lstm(FILENAME)
gc.collect()

# word2vec学習用のセンテンスを作成
def make_token_and_sentense(text: dict):
    page = len(text)
    # 単語のリスト一覧
    token_list = dict()
    word2vec_model_list = dict()
    
    for i in range(page):
        tmp = word_tokenize(text[i])
        token_list[i] = [tmp[j] for j in range(len(tmp)) if len(tmp[j]) > 3]
        
    sentense = [text[i].split() for i in range(page)]
    sentense_to_tfidf = [text[i] for i in range(page)]
    return token_list, sentense, sentense_to_tfidf

# 前処理後に残ったユニークな単語帳
def make_vocab(token_list: dict):
    page = len(token_list)
    vocab = list(set([token_list[i][j] for i in range(page) for j in range(len(token_list[i]))]))
    return vocab

# 作成したセンテンスからword2vecでモデル作成
def make_word2vec_model(sentense: list):
    model = word2vec.Word2Vec(sentense, vector_size=200, window=10, hs=1, min_count=2, sg=1)
    return model

token_list, sentense, sentense_to_tfidf = make_token_and_sentense(text)
vocab = make_vocab(token_list)
model = make_word2vec_model(sentense)
print('create model finished')
gc.collect()

# センテンスtf-idf計算
from sklearn.feature_extraction.text import TfidfVectorizer

def compute_tfidf(sentense_to_tfidf: list):
    tfidf_vectorizer = TfidfVectorizer()
    result = tfidf_vectorizer.fit_transform(sentense_to_tfidf)
    # 計算対象となった単語一覧取得
    feature_name = tfidf_vectorizer.get_feature_names()
    return result.toarray(), feature_name

result, feature_name = compute_tfidf(sentense_to_tfidf)
# result.toarray()
gc.collect()
