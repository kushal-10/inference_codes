from transformers import AutoProcessor, AutoModelForVision2Seq, AutoConfig
from PIL import Image
import requests
import torch

model_id = "google/paligemma-3b-mix-224"

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)

model = AutoModelForVision2Seq.from_pretrained(model_id).eval()
processor = AutoProcessor.from_pretrained(model_id)
model_config = AutoConfig.from_pretrained(model_id)

context = 0
# Some models have 'max_position_embeddings' others have - 'max_sequence_length' 
if hasattr(model_config, "text_config"):
    context = model_config.text_config.max_position_embeddings
    print("uses max position embeddings")
elif hasattr(model_config, "max_sequence_length"):
    context = model_config.max_sequence_lengths
    print("uses max sequence length")

# print(context) #8192 context

# Check chat template
chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

template_message = processor.tokenizer.apply_chat_template(chat, tokenize=False)
print(template_message)


# # Instruct the model to create a caption in Spanish
# prompt = "caption es"
# model_inputs = processor(text=prompt, images=image, return_tensors="pt")
# input_len = model_inputs["input_ids"].shape[-1]

# with torch.inference_mode():
#     generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
#     generation = generation[0][input_len:]
#     decoded = processor.decode(generation, skip_special_tokens=True)
#     print(decoded)
