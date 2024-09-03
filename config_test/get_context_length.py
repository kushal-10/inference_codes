from transformers import AutoConfig

def print_context_length(model_id):
    model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # Some models have 'max_position_embeddings', others have 'max_sequence_length'
    max_embeddings = getattr(model_config, 'max_position_embeddings', None) # Dolphin 72B
    if max_embeddings is not None:
        context = max_embeddings
    else:
        text_config = getattr(model_config, 'text_config', None) # Idefics3
        if not text_config:
            text_config = getattr(model_config, 'llm_config') # InternVL2
        context = getattr(text_config, 'max_position_embeddings')

    print(f"Context limit for model - {model_id} is {context}")\
    
model_ids = ["Phi-3-vision-128k-instruct", "Phi-3.5-vision-instruct", "Idefics3-8B-Llama3","internlm-xcomposer2d5-7b", "InternVL2-8B", "InternVL2-26B",
             "InternVL2-40B", "InternVL2-Llama3-76B"]

for model_id in model_ids:
    print_context_length(model_id)
