from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "monologg/bert-base-cased-goemotions-original"
save_path = "./emotion_model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
