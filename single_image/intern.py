import torch
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

# Define the CUDA devices you want to attempt loading on
cuda_devices = [1, 2, 3]  # CUDA devices 1, 2, 3

# Initialize model and tokenizer
model = None
for device_id in cuda_devices:
    try:
        model = AutoModel.from_pretrained('internlm/internlm-xcomposer2d5-7b',
                                          device_map={'cuda:0': f'cuda:{device_id}'},  # Specify the device map
                                          torch_dtype=torch.bfloat16,
                                          trust_remote_code=True).to(f'cuda:{device_id}').eval()
        tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2d5-7b',
                                                  device_map={'cuda:0': f'cuda:{device_id}'},  # Specify the device map
                                                  trust_remote_code=True)
        model.tokenizer = tokenizer
        break  # If successful, break out of the loop
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"Failed to allocate memory on CUDA device {device_id}. Trying next device...")
            continue
        else:
            raise e  # Re-raise if the error is not related to out of memory

if model is None:
    raise RuntimeError("Failed to load the model on any available CUDA device.")

# Example queries and images for inference
query1 = 'Image1 <ImageHere>; Image2 <ImageHere>; Image3 <ImageHere>; I want to buy a car from the three given cars, analyze their advantages and weaknesses one by one'
query2 = 'Image4 <ImageHere>; How about the car in Image4'
images = ['./examples/cars1.jpg',
          './examples/cars2.jpg',
          './examples/cars3.jpg',
          './examples/cars4.jpg']

# Inference loop
history = None
for query, image_path in zip([query1, query2], images):
    # Perform inference with autocast and device placement
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        response, history = model.chat(tokenizer, query, images, do_sample=False, num_beams=3, history=history, use_meta=True)
    print(response)
