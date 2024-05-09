# internlm/internlm-xcomposer2-vl-7b - Chat support
# pip install torchvision -> Requires torch - 2.2.0

import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor
import requests
from PIL import Image
import os

# torch.set_grad_enabled(False)


image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

# init model and tokenizer
model = AutoModel.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True)

query = '<ImageHere>Please describe this image in detail.'
# image = './image1.webp'
with torch.cuda.amp.autocast():
  response, _ = model.chat(tokenizer, query=query, image=image1, history=[], do_sample=False)
print(response)
#The image features a quote by Oscar Wilde, "Live life with no excuses, travel with no regret,"
# set against a backdrop of a breathtaking sunset. The sky is painted in hues of pink and orange,
# creating a serene atmosphere. Two silhouetted figures stand on a cliff, overlooking the horizon.
# They appear to be hiking or exploring, embodying the essence of the quote.
# The overall scene conveys a sense of adventure and freedom, encouraging viewers to embrace life without hesitation or regrets.
