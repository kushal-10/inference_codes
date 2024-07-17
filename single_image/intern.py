import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from PIL import Image
import requests

torch.set_grad_enabled(False)

# init model and tokenizer
model = AutoModelForCausalLM.from_pretrained('internlm/internlm-xcomposer2d5-7b', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2d5-7b', trust_remote_code=True)
model.tokenizer = tokenizer



config = AutoConfig.from_pretrained('internlm/internlm-xcomposer2d5-7b')
print(config)

#
# #### CUSTOM MAPWORLD EXAMPLE
msg = [{'role': 'user', 'content': 'We are currently in this room. Please help me with the following task. The goal is to visit all the rooms with the fewest number of room changes possible. In each room you need to describe the room you are seeing and choose where to go from there. Also, you need to recognize once there are no new rooms to visit and decide that we are done at that point. Please give your answer in the following format: "{"description": "<room description>", "action": "<action>"}". Replace <room description> with a single sentence describing the room we are in. To move to a neighboring room, replace <action> with "GO: DIRECTION" where DIRECTION can be one of [north, south, east, west]. To stop the exploration, replace <action> with "DONE". Omit any other text.\nHere is an example:\nWe are in this room. From here we can go: north, west. What is your next instruction?\n{"description": "We are in a kitchen with a red fridge.", "action": "GO: north"}\nWe have made a step and are now in this room. From here we can go: south, east. What is your next instruction?\n{"description": "We are in a living room with a couch and a tv.", "action": "GO: east"}\n...\nWe have made a step and are now in this room. From here we can go: south, east. What is your next instruction?\n{"description": "We are in a bathroom", "action": "DONE"}\nLet us start. \nWe have made a step and are now in this room. From here we can go: west. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a bedroom with a bunk bed and a window.", "action": "GO: west"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: south, east. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a room with a door to the south and a door to the east.", "action": "GO: east"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: west. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a bedroom with a bunk bed and a window.", "action": "GO: west"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: south, east. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a room with a door to the south and a door to the east.", "action": "GO: east"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: west. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a bedroom with a bunk bed and a window.", "action": "GO: west"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: south, east. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a room with a door to the south and a door to the east.", "action": "GO: east"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: west. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a bedroom with a bunk bed and a window.", "action": "GO: west"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: south, east. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a room with a door to the south and a door to the east.", "action": "GO: east"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: west. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a bedroom with a bunk bed and a window.", "action": "GO: west"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: south, east. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a room with a door to the south and a door to the east.", "action": "GO: east"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: west. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a bedroom with a bunk bed and a window.", "action": "GO: west"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: south, east. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a room with a door to the south and a door to the east.", "action": "GO: east"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: west. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a bedroom with a bunk bed and a window.", "action": "GO: west"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: south, east. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a room with a door to the south and a door to the east.", "action": "GO: east"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: west. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a bedroom with a bunk bed and a window.", "action": "GO: west"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: south, east. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a room with a door to the south and a door to the east.", "action": "GO: east"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: west. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a bedroom with a bunk bed and a window.", "action": "GO: west"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: south, east. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a room with a door to the south and a door to the east.", "action": "GO: east"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: west. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a bedroom with a bunk bed and a window.", "action": "GO: west"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: south, east. What is your next instruction?', 'image': []}]

history = []
image = []
image_counter = 0
for m in msg:
    if m['role'] == 'user':
        prev_user_msg = m['content']
        if 'image' in m:
            if type(m['image']) == str:
                # A single image is passed
                image_counter += 1
                prev_user_msg = f"Image{image_counter} <ImageHere>; " + prev_user_msg
                image.append(m['image'])
            elif type(m['image']) == list:
                # A list of images is passed
                for img in m['image']:
                    image_counter += 1
                    prev_user_msg = f"Image{image_counter} <ImageHere>; " + prev_user_msg
                    image.append(img)
            else:
                print("Please pass a valid value of image in the message - Either a str or List[str]")

    elif m['role'] == 'assistant':
        # Append User+Assistant Message in Sequence
        history.append((prev_user_msg, m['content']))

curr_message = prev_user_msg


query = "What do you see in this image?"
image_link = "https://cs.stanford.edu/people/rak248/VG_100K/2353739.jpg"
image = Image.open(requests.get(image_link, stream=True).raw).convert('RGB')


# query = curr_message
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response, his = model.chat(tokenizer, query, image, do_sample=False, top_p=1, num_beams=3, history=history, use_meta=True)
    # Unset top_p manually to avoid the following warning
    #  UserWarning: `do_sample` is set to `False`. However, `top_p` is set to `0` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.
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
