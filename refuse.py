import os
import json
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import *


DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

class CustomDataset(Dataset):
    def __init__(self, data, system_prompt):
        self.data = data
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        question = row["question"]
        matching_option = row["answer_matching_behavior"]
        not_matching_option = row["answer_not_matching_behavior"]
        matching_option_tokens = get_prompt_tokens(self.tokenizer, self.system_prompt, question, matching_option)
        not_matching_option_tokens = get_prompt_tokens(self.tokenizer, self.system_prompt, question, not_matching_option)
        return matching_option_tokens, not_matching_option_tokens

class BlockOutputWrapper(torch.nn.Module):
    def __init__(self, block, tokenizer):
        super().__init__()
        self.block = block
        self.tokenizer = tokenizer

        self.activations = None
        self.activation_to_add = None
        self.add_after_position = None

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]
        if self.activation_to_add is not None:
            steered_activation = self.add_activation_after_position(activation_to_add=self.activation_to_add, position_ids=kwargs["position_ids"])
            output = (steered_activation + self.activation_to_add,) + output[1:]

        return output

    def add_activation_after_position(self, activation_to_add, position_ids):
        if self.add_after_position is None:
            self.add_after_position = position_ids.min().item() - 1
        mask = position_ids > self.add_after_position
        mask = mask.unsqueeze(-1)
        steered_activation = self.activations + (mask.float() * activation_to_add)
        return steered_activation

    def set_activation_to_add(self, activation_to_add):
        self.activation_to_add = activation_to_add

    def reset(self):
        self.activation_to_add = None
        self.activations = None
        self.add_after_position = None

class Llama2_Chat_7B_Wrapper:
    def __init__(self, system_prompt):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf").to(self.device)
        self.E_INST = torch.tensor(self.tokenizer.encode("[/INST]")[1:]).to(self.device)

        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(layer, self.tokenizer)

    def set_add_after_position(self, position):
        for layer in self.model.model.layers:
            layer.add_after_position = position

    def generate_text(self, prompt, max_new_tokens=100):
        tokens = get_prompt_tokens(self.tokenizer, self.system_prompt, prompt).to(self.device)
        return self.generate(tokens, max_new_tokens=max_new_tokens)

    def generate(self, tokens, max_new_tokens=100):
        e_inst_tok_position = find_instruction_end_postion(tokens[0], self.E_INST)
        self.set_add_after_position(e_inst_tok_position)
        generated = self.model.generate(inputs=tokens, max_new_tokens=max_new_tokens, top_k=1)
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    def get_logits(self, tokens):
        with torch.no_grad():
            logits = self.model(tokens).logits
            return logits

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].activations

    def set_activation_to_add(self, layer, activation_to_add):
        self.model.model.layers[layer].set_activation_to_add(activation_to_add)

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

if __name__ == "__main__":
    data_path = "refusal_data_A_B.json"

    with open(data_path, "r") as f:
        data = json.load(f)

    dataset = CustomDataset(data, DEFAULT_SYSTEM_PROMPT)
    model = Llama2_Chat_7B_Wrapper(DEFAULT_SYSTEM_PROMPT)

    start_layer = 0
    end_layer = 31

    # gen_steering_vectors(model, dataset, start_layer=start_layer, end_layer=end_layer)

    questions = ["Do you enjoy action movies?"]

    layers = [10, 12, 14, 16]
    multiplier_list = [x / 10 for x in range(-32, 32, 4)]
    max_new_tokens = 300

    all_results = {}
    dc = {}

    for question in tqdm(questions, desc="question", position=0):
        dc_t = dc.copy()
        for layer in tqdm(layers, desc="layer", position=1):
            dc_t[layer] = {}
            for multiplier in multiplier_list:
                model.reset_all()
                steering_vector = get_steering_vector(layer)
                model.set_activation_to_add(layer, multiplier * steering_vector.cuda())
                answer = model.generate_text(question, max_new_tokens=max_new_tokens)
                answer = answer.split("[/INST]")[-1].strip()
                dc_t[layer][multiplier] = answer
        all_results[question] = dc_t

    with open("results.json", "w") as f:
        json.dump(all_results, f)