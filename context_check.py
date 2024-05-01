from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig

model_id = "llava-hf/llava-1.5-7b-hf"

model_config = AutoConfig.from_pretrained(model_id)
print(model_config)
print(model_config.text_config.max_position_embeddings)