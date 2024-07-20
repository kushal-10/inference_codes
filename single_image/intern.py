import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from PIL import Image
import requests
import torchvision.transforms.functional as F

torch.set_grad_enabled(False)


def custom_padding(img, padding):
    num_channels = len(img.getbands())
    if num_channels == 4:  # RGBA
        fill_color = (255, 255, 255, 255)
    elif num_channels == 3:  # RGB
        fill_color = (255, 255, 255)
    else:
        raise ValueError(f"Unsupported number of channels: {num_channels}")

    padded_img = F.pad(img, padding, fill=fill_color)
    return padded_img


def pad_image(image_path, padding):
    img = Image.open(image_path)
    padded_img = custom_padding(img, padding)
    if padded_img.mode == 'RGBA':
        padded_img = padded_img.convert('RGB')  # Convert RGBA to RGB
    return padded_img


model = AutoModel.from_pretrained('internlm/internlm-xcomposer2d5-7b', torch_dtype=torch.bfloat16,
                                  trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2d5-7b', trust_remote_code=True)
model.tokenizer = tokenizer

query = 'Image1 <ImageHere>; What is shown in this image?'
image_path = './examples/8.jpg'
image = [image_path]

# Define the padding size
padding = (0, 0, 560, 336)

# Pad the image
padded_img = pad_image(image_path, padding)
padded_img.save('padded_image.jpg')  # Save the image in RGB mode

with torch.autocast(device_type='cuda', dtype=torch.float16):
    response, his = model.chat(tokenizer, query, ['padded_image.jpg'], do_sample=False, num_beams=3, use_meta=True)
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
