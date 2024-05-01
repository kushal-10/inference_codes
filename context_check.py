from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig

model_id = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(model_id, use_fast=False, device_map="auto", verbose=False)
model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

model_config = AutoConfig.from_pretrained(model_id)
print(model_config.max_position_embeddings)