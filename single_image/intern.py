import torch
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

# init model and tokenizer
model = AutoModel.from_pretrained('internlm/internlm-xcomposer2d5-7b', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2d5-7b', trust_remote_code=True)
model.tokenizer = tokenizer


# #### CUSTOM MAPWORLD EXAMPLE
msg = [{'role': 'user', 'content': 'We are currently in this room. Please help me with the following task. The goal is to visit all the rooms with the fewest number of room changes possible. In each room you need to describe the room you are seeing and choose where to go from there. Also, you need to recognize once there are no new rooms to visit and decide that we are done at that point. Please give your answer in the following format: "{"description": "<room description>", "action": "<action>"}". Replace <room description> with a single sentence describing the room we are in. To move to a neighboring room, replace <action> with "GO: DIRECTION" where DIRECTION can be one of [north, south, east, west]. To stop the exploration, replace <action> with "DONE". Omit any other text.\nHere is an example:\nWe are in this room. From here we can go: north, west. What is your next instruction?\n{"description": "We are in a kitchen with a red fridge.", "action": "GO: north"}\nWe have made a step and are now in this room. From here we can go: south, east. What is your next instruction?\n{"description": "We are in a living room with a couch and a tv.", "action": "GO: east"}\n...\nWe have made a step and are now in this room. From here we can go: south, east. What is your next instruction?\n{"description": "We are in a bathroom", "action": "DONE"}\nLet us start. \nWe have made a step and are now in this room. From here we can go: west. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a bedroom with a bunk bed and a window.", "action": "GO: west"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: south, east. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a room with a door to the south and a door to the east.", "action": "GO: east"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: west. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a bedroom with a bunk bed and a window.", "action": "GO: west"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: south, east. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a room with a door to the south and a door to the east.", "action": "GO: east"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: west. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a bedroom with a bunk bed and a window.", "action": "GO: west"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: south, east. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a room with a door to the south and a door to the east.", "action": "GO: east"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: west. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a bedroom with a bunk bed and a window.", "action": "GO: west"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: south, east. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a room with a door to the south and a door to the east.", "action": "GO: east"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: west. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a bedroom with a bunk bed and a window.", "action": "GO: west"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: south, east. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a room with a door to the south and a door to the east.", "action": "GO: east"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: west. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a bedroom with a bunk bed and a window.", "action": "GO: west"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: south, east. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a room with a door to the south and a door to the east.", "action": "GO: east"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: west. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a bedroom with a bunk bed and a window.", "action": "GO: west"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: south, east. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a room with a door to the south and a door to the east.", "action": "GO: east"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: west. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a bedroom with a bunk bed and a window.", "action": "GO: west"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: south, east. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a room with a door to the south and a door to the east.", "action": "GO: east"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: west. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a bedroom with a bunk bed and a window.", "action": "GO: west"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: south, east. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a room with a door to the south and a door to the east.", "action": "GO: east"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: west. What is your next instruction?'}, {'role': 'assistant', 'content': '{"description": "We are in a bedroom with a bunk bed and a window.", "action": "GO: west"}'}, {'role': 'user', 'content': 'We have made a step and are now in this room. From here we can go: south, east. What is your next instruction?', 'image': ['examples/ADE_train_00016739.jpg']}]
# msg = [{'role': 'user', 'content': 'Halloe akfawrgshs fneffivprs', 'image': 'https://temp.img'}]
history = []
image = []
image_counter = 0
for m in msg:
    if m['role'] == 'user':
        image_counter += 1
        prev_user_msg = m['content']
        if 'image' in m:
            if type(m['image']) == str:
                # A single image is passed
                prev_user_msg = f"Image{image_counter} <ImageHere>; " + prev_user_msg
                image.append(m['image'])
            elif type(m['image']) == list:
                # A list of images is passed
                for img in m['image']:
                    prev_user_msg = f"Image{image_counter} <ImageHere>; " + prev_user_msg
                    image.append(img)
                    image_counter += 1
                image_counter -= 1
            else:
                print("Please pass a valid value of image in the message - Either a str or List[str]")

    elif m['role'] == 'assistant':
        # Append User+Assistant Message in Sequence
        history.append((prev_user_msg, m['content']))

curr_message = prev_user_msg

# print(len(history), curr_message)
# print(image, image_counter)

# Image number [posiiton in message ] should reflect in Placeholder

query = curr_message
image.append('./examples/ADE_train_00016739.jpg')
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response, his = model.chat(tokenizer, query, image, do_sample=False, top_p=0, num_beams=3, history=history, use_meta=True)
print(response)




# his = [('Image1 <ImageHere>; I want to buy a car from the three given cars, analyze their advantages and weaknesses one by one. This is the first car.', "To proceed, I need the images of the three cars you're referring to. Please upload the images, and then I will be able to provide a detailed analysis of each car's advantages and disadvantages."), ('Image2 <ImageHere>; This is the second car.', "To proceed, I need the image of the second car you're referring to. Please upload the image, and then I will be able to provide a detailed analysis of its advantages and disadvantages."), ('Image3 <ImageHere>; This is the third car.', "To proceed, I need the image of the third car you're referring to. Please upload the image, and then I will be able to provide a detailed analysis of its advantages and disadvantages.")]

# print(type(his[0]))