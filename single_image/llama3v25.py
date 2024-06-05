# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoModelForVision2Seq, AutoConfig
import requests

model = AutoModelForVision2Seq.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
model = model.to(device='cuda')

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
model.eval()

model_config = AutoConfig.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
if hasattr(model_config, "text_config"):
    context = model_config.text_config.max_position_embeddings
elif hasattr(model_config, "max_sequence_length"):
    context = model_config.max_sequence_length
else:
    context = 10

print(f"Context : {context}")


image = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
image1 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

question = 'What is in the image?'
msgs = [
    {'role': 'user', 'content': 'How are these two images different?', 'image': ['https://llava-vl.github.io/static/images/view.jpg', 'http://images.cocodataset.org/val2017/000000039769.jpg']}
]

res = model.chat(
    image=[image, image1],
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=True, # if sampling=False, beam_search will be used by default
    temperature=0.5,
    mmax_new_tokens=100,
    # system_prompt='' # pass system_prompt if needed
)
print(res)

'''
NOTES : 
1) Does accept system message, but needs to be passed separately

2) Accepts multiple images but bad response

3) Doesnot work when temperature is strictly 0

4) Major Hallucinations - Given images 1) lake and 2) two cats
Output with t=0.0001
The two images are different in several ways:

1. Content: The first image shows a person holding a smartphone, while the second image shows a person holding a tablet.

2. Device: The first image features a smartphone, which is typically smaller and more portable than a tablet. The second image features a tablet, which is larger and often used for tasks that require more screen space.

3. Screen size: The first image shows a smaller screen, while the second image shows a larger screen.

4. Usage: Smartphones are often used for communication, social media, and other personal activities, while tablets are often used for work, entertainment, and other productivity tasks.

5. Design: The design of the devices may also differ, with smartphones typically having a more compact and sleek design, while tablets may have a larger and more robust design.

Overall, these two images depict different types of devices and their respective uses, highlighting the diversity of technology available to us today.


5) Try with a higher temperature - same response but in a paragraph, not points

5.1) Hallucinations are real bad - 
The two images are different in their content and context. The first image is a painting of a man with a mustache, while the second image is a photograph of a person with a mustache. The painting is an artistic representation, while the photograph is a real-life depiction. Additionally, the style and technique used to create the images differ, with the painting likely being created using traditional art materials such as oil or acrylic paint, and the photograph being taken with a camera.


6) Try in backend anyway....
'''