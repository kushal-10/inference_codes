# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import requests

model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
model = model.to(device='cuda')

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
model.eval()

image = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
question = 'What is in the image?'
msgs = [{'role': 'user', 'content': question, 'image': "https://llava-vl.github.io/static/images/view.jpg"}]

res = model.chat(
    image=image,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True, # if sampling=False, beam_search will be used by default
    temperature=0.7,
    # system_prompt='' # pass system_prompt if needed
)
print(res)