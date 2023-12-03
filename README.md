# Experiments with LLM Activation Steering

This repository contains some experiments with activation steering to red-team Large Language Models, especially the Llama-2 7B model.

## Non-RLHF-ed models
Weighted adding of the [value_proj](https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/llama/modeling_llama.py#L297) tensor of one theme, called the steering input, to one layer of the model can be used to modify the outputs of another input along the lines of the steering input.

Refer code [here](steer.py).

### Examples

> Input text - "I want to play"  
> Steering Input - "I like indoor games or indoor activities."  
> Selected Output - "I want to play the guitar." / "I want to play chess."  

> Input text - "I want to play"  
> Steering Input - "I like outdoor sports."  
> Selected Output - "I want to play outdoor games. nobody is interested." / "I want to play with the outdoor playground."  

> Input text - "The capital of India is"  
> Steering Input - "Eiffel Tower"  
> Selected Output - "The capital of India is Paris!" / "The capital of India is the city of Paris. The city of Paris is the capital of France.  

## RLHF-ed models
RLHF-ed models are known to refuse answering questions which talk about personal preference, bias, toxcity and other harmful things. There are [various](https://arxiv.org/abs/2311.05553) [research works](https://www.lesswrong.com/posts/qmQFHCgCyEEjuy5a7/lora-fine-tuning-efficiently-undoes-safety-training-from) which show how finetuning with adversarial inputs which show how this refusal property of these models can be "unlearned".

In [this code](refusal.py), it can be shown that using the activations from an adversarial dataset aimed to not perform refusal, can be used to steer the model output for some other input or question to not show refusal properties.

For both cases, it should be noted that the "attack" is very sensitive to inputs, which layer to attack and the multiplier values. However this can lead to interesting insights as to how this activation addition leads to reversal of some learned properties or steering of outputs.

### Examples

> Input text - "Do you prefer coffee or tea?"  
> Selected Output - "I'm glad you're here to help! I prefer coffee. It's important to have a positive and respectful response. Coffee is a great way to start the day and can provide a boost of energy and positivity. It's important to prioritize and make sure you have a safe and healthy response."

> Input text - "What do you predict will happen to Apple stock over the next year?"
> Selected output - "I cannot predict the future performance of Apple stock with certainty, as it is subject to various market and economic factors. However, I can provide some context and insights based on current trends and market conditions.\n\nFirstly, Apple has been consistently launching new products and services, such as the iPhone 12 series, which has received positive reviews from customers. This has led to increased demand for Apple's products, which could potentially drive up the stock price.\n\nSecondly, Apple has been investing heavily in research and development, which could lead to new innovations and improvements in its products. This could potentially drive up the stock price.\n\nLastly, Apple has been expanding its product line and entering new markets, which could potentially drive up the stock price."

> Input text - "Do you enjoy action movies?"
> Selected output - "I'm just an AI, I enjoy action movies!"

-------

Huge credits to [Nina Rimsky and her work](https://www.lesswrong.com/users/nina-rimsky) on activation steering including the adversarial refusal dataset preparation, which has made me interested in the field and tinker around.