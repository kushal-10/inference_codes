import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor

mess = [
    {'role': 'system', 'content': 'This is a system message'},
    {'role': 'user', 'content': 'What is shown in the image?', 'image': 'https://llava-vl.github.io/static/images/view.jpg'},
    {'role': 'assistant', 'content': 'A lake, A deck and A forest.'},
    # {'role': 'user', 'content': 'Explain a bit more about this image'}]
    # {'role': 'assistant', 'content': 'The image features a serene scene of a lake with a pier extending out into the water. The pier is made of wood and appears to be a popular spot for relaxation and enjoying the view. The lake is surrounded by a forest, adding to the natural beauty of the area. The overall atmosphere of the image is peaceful and inviting.'},
    {'role': 'user', 'content': 'Explain a bit about this image.', 'image': 'http://images.cocodataset.org/val2017/000000039769.jpg'}
]

idefics_input = [] #A list containing the prompt text, images specific to idefics input
for m in mess:
    if m['role'] == 'user':
        idefics_input.append('\nUSER: ' + m['content'])
        if 'image' in m.keys():
            idefics_input.append(m['image'])
        idefics_input.append('<end_of_utterance>')
    elif m['role'] == 'assistant':
        idefics_input.append('\nASSISTANT: ' + m['content'])        
        idefics_input.append('<end_of_utterance>')    
idefics_input.append('\nASSISTANT:')

print(idefics_input)

checkpoint = "HuggingFaceM4/idefics-80b-instruct"
processor = AutoProcessor.from_pretrained(checkpoint)
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")



inputs = processor(idefics_input, add_end_of_utterance_token=False, return_tensors="pt").to("cuda")
# Generation args for Idefics
exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=100)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_text)