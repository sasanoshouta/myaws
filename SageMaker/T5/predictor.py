# predictor.py
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

tokenizer = T5Tokenizer.from_pretrained(model_dir_path)
trained_model = T5ForConditionalGeneration.from_pretrained(model_dir_path)

generated_texts = generate_text_from_model(title=title, trained_model=trained_model, tokenizer=tokenizer, num_return_sequences=10)
