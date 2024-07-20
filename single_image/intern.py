import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from PIL import Image
import requests

torch.set_grad_enabled(False)

# init model and tokenizer
model = AutoModel.from_pretrained('internlm/internlm-xcomposer2d5-7b', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2d5-7b', trust_remote_code=True)
model.tokenizer = tokenizer

query = 'Image1 <ImageHere>; What is shown in this image?'
image_path = './examples/8.jpg'
image = [image_path]

# Open the image to check the number of channels
img = Image.open(image_path)
num_channels = len(img.getbands())

# Choose the fill color based on the number of channels
if num_channels == 4:  # RGBA
    fill_color = [255, 255, 255, 255]
elif num_channels == 3:  # RGB
    fill_color = [255, 255, 255]
else:
    raise ValueError(f"Unsupported number of channels: {num_channels}")

with torch.autocast(device_type='cuda', dtype=torch.float16):
    response, his = model.chat(tokenizer, query, image, do_sample=False, num_beams=3, use_meta=True, fill=fill_color)
print(response)



"""
NOTES:
 Requires additional - torchvision==0.15.2 -> Change to torch 0.16.1 after torch updae from 2.0.1 to 2.1.1
 decord==0.6.0
 
 Need to supress this warning - /project/kkoshti/testing/clem/lib/python3.10/site-packages/huggingface_hub/file_download.py:1132: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
  warnings.warn(
And this:
/project/kkoshti/testing/clem/lib/python3.10/site-packages/torch/_utils.py:776: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage will be the only storage class. This should only matter to you if you are using storages directly.  To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()
  return self.fget.__get__(instance, owner)()


"""
