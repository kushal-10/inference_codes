from transformers import AutoConfig

modelids = ["HuggingFaceM4/idefics-9b-instruct",
            "llava-hf/llava-v1.6-vicuna-7b-hf",
            "llava-hf/llava-1.5-13b-hf"]

for id in modelids:
    model_config = AutoConfig.from_pretrained(id, token="randomtoken")
    print(model_config)
