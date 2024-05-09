from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig
from typing import Dict


model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
tokenizer = processor.tokenizer

prompt = "USER: <image> What is shown in this image? \nASSISTANT:"
tokenized_prompt = tokenizer.tokenize(prompt)


print(tokenized_prompt)
print(len(tokenized_prompt))