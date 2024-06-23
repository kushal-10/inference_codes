import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor

from jinja2 import Template

# We feed to the model an arbitrary sequence of text strings and images. Images can be either URLs or PIL Images.
prompts = [
    [
        "User: What is in this image?",
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "<end_of_utterance>",

        "User: What is in this image?",
                "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
                "<end_of_utterance>",

        "User: What is in this image?",
                "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
                "<end_of_utterance>",

        "User: What is in this image?",
                "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
                "<end_of_utterance>",

        "\nAssistant: This picture depicts Idefix, the dog of Obelix in Asterix and Obelix.<end_of_utterance>",

        "\nUser:",
        "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052",
        "And who is that?<end_of_utterance>",

        "\nAssistant:",
    ],
]

# prompt = '''
# User: What is shown in the image? <end_of_uttereance_>
# '''

# promtttt = ['''
# User: What is in this image?<end_of_utterance>
# Assistant: This picture depicts Idefix, the dog of Obelix in Asterix and Obelix.<end_of_utterance>
# User: who is that?<end_of_utterance>
# Assistant:
# ''' , 
# "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
# "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052"
# ]
# images_list = [
#     "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
#     "https://static.wikia.nocookie.net/asterix/images/2/25/R22b.gif/revision/latest?cb=20110815073052"
# ]

# input_thingy = prompt + images_list

# imagee = "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG"

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "HuggingFaceM4/idefics-9b-instruct"
processor = AutoProcessor.from_pretrained(checkpoint)
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")


# --batched mode
inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
# --single sample mode
# inputs = processor(prompts[0], return_tensors="pt").to(device)

# Generation args
exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_length=100)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
for i, t in enumerate(generated_text):
    print(f"{i}:\n{t}\n")


