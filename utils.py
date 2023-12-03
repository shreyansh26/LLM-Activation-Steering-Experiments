from tqdm import tqdm
from collections import defaultdict

import torch

def pad_tensors_to_same_size(tensor1, tensor2):
    # Ensure tensor2 is no larger than tensor1 along the second dimension
    if tensor2.size(1) > tensor1.size(1):
        tensor2 = tensor2[:, :tensor1.size(1), :]

    # In case tensor2 is smaller, pad it with zeros to match tensor1's size
    padding_size2 = max(0, tensor1.size(1) - tensor2.size(1))
    if padding_size2 > 0:
        padding2 = torch.zeros((tensor2.size(0), padding_size2, tensor2.size(2)), device=tensor2.device)
        tensor2 = torch.cat([tensor2, padding2], dim=1)

    return tensor1, tensor2

def get_prompt_tokens(tokenizer, system_prompt, instruction, output=None):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    if output:
        prompt = f"{B_INST} {B_SYS}{system_prompt}{E_SYS}{instruction.strip()} {E_INST} {output.strip()}"
    else:
        prompt = f"{B_INST} {B_SYS}{system_prompt}{E_SYS}{instruction.strip()} {E_INST}"

    return torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

def find_subtensor_position(tensor, sub_tensor):
    n, m = tensor.size(0), sub_tensor.size(0)
    if m > n:
        return -1
    for i in range(n - m + 1):
        if torch.equal(tensor[i : i + m], sub_tensor):
            return i
    return -1

def find_instruction_end_postion(tokens, end_inst):
    end_pos = find_subtensor_position(tokens, end_inst)
    return end_pos + len(end_inst) - 1

def gen_steering_vectors(model, dataset, start_layer=0, end_layer=31, token_idx=-2):
    # token_idx here is -2 to indicate the option A/B, as originally in the dataset, they are suffixed by a parathesis - (A), (B)
    layers = list(range(start_layer, end_layer + 1))
    positive_activations = defaultdict(list)
    negative_activations = defaultdict(list)

    for matching_option_tokens, not_matching_option_tokens in tqdm(dataset, desc="Processing prompts"):
        matching_option_tokens = matching_option_tokens.to(model.device)
        not_matching_option_tokens = not_matching_option_tokens.to(model.device)

        model.reset_all()
        model.get_logits(matching_option_tokens)
        for layer in layers:
            positive_activation = model.get_last_activations(layer)
            positive_activation = positive_activation[0, token_idx, :].detach().cpu()
            positive_activations[layer].append(positive_activation)

        model.reset_all()
        model.get_logits(not_matching_option_tokens)
        for layer in layers:
            negative_activation = model.get_last_activations(layer)
            negative_activation = negative_activation[0, token_idx, :].detach().cpu()
            negative_activations[layer].append(negative_activation)

    for layer in layers:
        positive_activation = torch.stack(positive_activations[layer])
        negative_activation = torch.stack(negative_activations[layer])
        steering_vector = (positive_activation - negative_activation).mean(dim=0)
        torch.save(steering_vector, f"vectors/vec_layer_{layer}.pt")

def get_steering_vector(layer):
    return torch.load(f"vectors/vec_layer_{layer}.pt")