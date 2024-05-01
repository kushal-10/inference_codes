from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig

# model_ids = [
#     "llava-hf/llava-1.5-7b-hf",
#     "llava-hf/llava-1.5-13b-hf",
#     "HuggingFaceM4/idefics-80b-instruct",
#     "llava-hf/llava-v1.6-vicuna-7b-hf",
#     "llava-hf/llava-v1.6-vicuna-13b-hf",
#     "llava-hf/llava-v1.6-mistral-7b-hf",
#     "llava-hf/llava-v1.6-34b-hf"]

# for model in model_ids:
#     model_config = AutoConfig.from_pretrained(model)
#     # print(model_config)
#     print(f"Max embeddings for {model}: {model_config.text_config.max_position_embeddings}")

model_id = "HuggingFaceM4/idefics-80b-instruct"
model_config = AutoConfig.from_pretrained(model)
print(model_config)