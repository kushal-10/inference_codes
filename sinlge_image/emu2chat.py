from PIL import Image
import requests
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq


image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
# image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
# image1 = "https://llava-vl.github.io/static/images/view.jpg"
# image2 = "http://images.cocodataset.org/val2017/000000039769.jpg"


# Requires protobuf, sentencepiece, timm, xformers

'''
constants.py: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 1.07k/1.07k [00:00<00:00, 4.98MB/s]
A new version of the following files was downloaded from https://huggingface.co/BAAI/Emu2-Chat:
- constants.py
. Make sure to double-check they do not contain any added malicious code. To avoid downloading new versions of the code file, you can pin a revision.
A new version of the following files was downloaded from https://huggingface.co/BAAI/Emu2-Chat:
- visual.py
- constants.py, atleast 95GB VRAM
'''

tokenizer = AutoTokenizer.from_pretrained("BAAI/Emu2-Chat", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("BAAI/Emu2-Chat", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("BAAI/Emu2-Chat", device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).to('cuda')


# '''
# With tokenizer
# '''
# # `[<IMG_PLH>]` is the image placeholder which will be replaced by image embeddings. 
# # the number of `[<IMG_PLH>]` should be equal to the number of input images

# # device_map = infer_auto_device_map(model, max_memory={0:'11GiB',1:'11GiB', 2:'11GiB',}, no_split_module_classes=['Block','LlamaDecoderLayer'])  
# # # input and output logits should be on same device
# # device_map["model.decoder.lm.lm_head"] = 0

# # model = load_checkpoint_and_dispatch(
# #     model, 
# #     '../../data/huggingface/models--BAAI--Emu2-Chat',
# #     device_map=device_map).eval()


# query = '[<IMG_PLH>]Describe the image in detail:' 

# inputs = model.build_input_ids(
#     text=[query],
#     tokenizer=tokenizer,
#     image=[image1]
# )

# # with torch.no_grad():
# outputs = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], image=inputs["image"].to(torch.bfloat16),
# max_new_tokens=64,
# length_penalty=-1)





# output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

# print(output_text)

# '''
# With Processor
# '''
prompt3 = '[<IMG_PLH>]Describe the image in detail:' 

# image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
# image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
image1 = "https://llava-vl.github.io/static/images/view.jpg"
# image2 = "http://images.cocodataset.org/val2017/000000039769.jpg"


inputs = processor(prompt3, image1, return_tensors="pt")

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200)

generated_text = processor.batch_decode(output, skip_special_tokens=True)
print(generated_text)