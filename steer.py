import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import *

class ValueProjWrapper(torch.nn.Module):
    def __init__(self, v_proj):
        super().__init__()
        self.v_proj = v_proj
        self.last_attn_values = None
        self.attn_values_to_add = None

    def forward(self, *args, **kwargs):
        output = self.v_proj(*args, **kwargs)
        self.last_attn_values = output
        if self.attn_values_to_add is not None:
            o1, o2 = pad_tensors_to_same_size(output, self.attn_values_to_add)
            return o1 + o2
        return output

    def set_attn_values_to_add(self, attn_values_to_add):
        self.attn_values_to_add = attn_values_to_add

    def reset(self):
        self.last_attn_values = None
        self.attn_values_to_add = None
    
class Llama2_7B_Wrapper:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(self.device)

        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i].self_attn.v_proj = ValueProjWrapper(layer.self_attn.v_proj)

    def generate_text(self, prompt, max_new_tokens=100):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generate_ids = self.model.generate(inputs.input_ids.to(self.device), max_new_tokens=max_new_tokens, top_k=1)
        return self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
    
    def get_logits(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
          logits = self.model(inputs.input_ids.to(self.device)).logits
          return logits

    def get_last_attn_values(self, layer):
        return self.model.model.layers[layer].self_attn.v_proj.last_attn_values

    def set_attn_values_to_add(self, layer, attn_values_to_add):
        self.model.model.layers[layer].self_attn.v_proj.set_attn_values_to_add(attn_values_to_add)

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.self_attn.v_proj.reset()

    def mix_activations(self, input_text, steering_input, multiplier, layer, max_new_tokens=100):
        self.reset_all()
        self.get_logits(steering_input)
        steering_values = None
        steering_values = self.get_last_attn_values(layer)
        steering_values *= multiplier
        self.set_attn_values_to_add(layer, steering_values)
        return self.generate_text(input_text, max_new_tokens=max_new_tokens)

if __name__ == "__main__":
    model = Llama2_7B_Wrapper()

    multiplier_list = [x for x in range(5, 35, 5)]
    layers = list(range(10, 28, 2))

    for layer in layers:
        for multiplier in multiplier_list:
            print(layer, multiplier)
            # print(model.mix_activations("I want to play", "I like indoor games or indoor activities.", multiplier, layer, max_new_tokens=50))
            print(model.mix_activations("I want to play", "I like outdoor sports.", multiplier, layer, max_new_tokens=50))
            # print(model.mix_activations("When I talk to people, I like to", "I am helpful, harmless and honest", multiplier, layer, max_new_tokens=50))
            # print(model.mix_activations("The capital of India is", "Eiffel Tower", multiplier, layer, max_new_tokens=50))