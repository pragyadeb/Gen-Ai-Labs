## Objective

The objective of this lab is to fine-tune a pre-trained generative model (GPT-2) for real-world applications. Students will fine-tune GPT-2 to build a Product Review Generator for e-commerce and a Recipe Instruction Generator for a food-tech application, learning how transfer learning adapts a general model to specific business domains.

## Learning Outcomes

After completing this lab, students will be able to:

1. Understand how fine-tuning applies to real-world industry applications
2. Load and configure a pre-trained GPT-2 model using Hugging Face Transformers
3. Prepare real-world domain-specific datasets for causal language modeling
4. Fine-tune the model and compare generated output before and after training
5. Evaluate the practical quality of generated product reviews and recipe instructions

## What is Fine-Tuning for Real-World Applications?

Fine-tuning takes a pre-trained model that already understands general language and trains it further on domain-specific data. This is how companies build AI-powered features without training models from scratch.

### Real-world examples of fine-tuning:

* E-Commerce: Generate product descriptions, customer reviews, and recommendation text
* Food-Tech: Generate recipe instructions, meal plans, and cooking tips from ingredient lists
* Healthcare: Generate clinical notes, patient summaries, and medical documentation
* Customer Support: Generate context-aware chatbot replies trained on company FAQ data
* Marketing: Generate ad copy, email campaigns, and social media posts in brand voice

---

# Experiment

## Component–I: Fine-Tune GPT-2 as a Product Review Generator (E-Commerce)

### Scenario

You are an AI engineer at an e-commerce company. The company wants to build an AI tool that auto-generates realistic product reviews to help sellers understand what good reviews look like and to populate demo storefronts. Your task is to fine-tune GPT-2 on real product review data.

### Tasks

1. Load GPT-2 and generate baseline product reviews (before fine-tuning)
2. Prepare the product review dataset and tokenize it
3. Fine-tune the model on product review data
4. Generate product reviews from the fine-tuned model and compare with baseline

### Dataset

Use the following product review corpus:

```
this phone has an amazing battery life and the camera quality is outstanding for the price.
i bought this laptop for college and it handles all my assignments and coding projects perfectly.
the sound quality of these headphones is incredible with deep bass and clear vocals.
this smartwatch tracks my steps accurately and the heart rate monitor is very reliable.
great wireless earbuds with noise cancellation that blocks out all background sound.

the keyboard feels very comfortable for long typing sessions and the backlight is a nice touch.
this portable charger saved me during travel and it charges my phone three times on a single charge.
the tablet screen is bright and colorful which makes watching movies a great experience.
i love this fitness tracker because it motivates me to reach my daily exercise goals.
this bluetooth speaker is compact but delivers surprisingly loud and clear audio.

the delivery was fast and the product was packed securely with no damage at all.
excellent value for money and the build quality feels premium despite the affordable price.
the customer service team was very helpful when i had questions about the product features.
this camera takes stunning photos in low light and the video recording quality is very smooth.
i have been using this product for three months and it still works perfectly like day one.

the design is sleek and modern and it looks great on my desk next to my other gadgets.
easy to set up right out of the box and the instructions were clear and simple to follow.
highly recommend this product to anyone looking for quality and reliability at a fair price.
the software updates keep adding new features which makes this purchase even more worthwhile.
best purchase i made this year and i would definitely buy from this brand again.
```

### Code

#### Step 1: Setup and Load Model

```
!pip install transformers datasets accelerate -q

import torch, math
from transformers import (GPT2LMHeadModel, GPT2Tokenizer, Trainer,
    TrainingArguments, DataCollatorForLanguageModeling, set_seed)
from datasets import Dataset
set_seed(42)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
```

#### Step 2: Generate Baseline Reviews (Before Fine-Tuning)

```
def generate_text(model, tokenizer, prompt, max_length=100):
    model.eval()
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(inputs, max_length=max_length, temperature=0.8,
            top_k=50, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0], skip_special_tokens=True)

review_prompts = [
    'This product is',
    'I bought this phone and',
    'The quality of this item',
]

print('=== BASELINE REVIEWS (Before Fine-Tuning) ===')
baseline = {}
for p in review_prompts:
    baseline[p] = generate_text(model, tokenizer, p)
    print(f'Prompt: {p}\nOutput: {baseline[p]}\n')
```

#### Step 3: Prepare Dataset and Fine-Tune

```
corpus = [
    'this phone has an amazing battery life and the camera quality is outstanding for the price.',
    ...
]

dataset = Dataset.from_dict({'text': corpus})
tokenized = dataset.map(lambda x: tokenizer(x['text'], truncation=True,
    max_length=128, padding='max_length'), batched=True, remove_columns=['text'])
split = tokenized.train_test_split(test_size=0.15, seed=42)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir='./gpt2-reviews', num_train_epochs=15,
    per_device_train_batch_size=4, learning_rate=5e-5,
    weight_decay=0.01, warmup_steps=50, eval_strategy='epoch',
    logging_steps=10, save_strategy='no',
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(model=model, args=training_args,
    train_dataset=split['train'], eval_dataset=split['test'],
    data_collator=data_collator)

trainer.train()
```

#### Step 4: Generate Reviews and Compare

```
eval_res = trainer.evaluate()
print(f'Perplexity: {math.exp(eval_res["eval_loss"]):.2f}')

print('\n=== FINE-TUNED REVIEWS (After Fine-Tuning) ===')
for p in review_prompts:
    ft_out = generate_text(model, tokenizer, p)
    print(f'Prompt: {p}')
    print(f'  Baseline:   {baseline[p][:120]}')
    print(f'  Fine-Tuned: {ft_out[:120]}\n')
```

### Expected Output

1. Baseline generates random generic text (news, Wikipedia-style content)
2. Fine-tuned model generates realistic product review language
3. Clear shift in tone, vocabulary, and structure

---

## Component–II: Fine-Tune GPT-2 as a Recipe Instruction Generator (Food-Tech)

### Scenario

You are an AI developer at a food-tech startup. The company is building a smart cooking app that suggests recipes. Your task is to fine-tune GPT-2 to generate cooking instructions when given a dish name as a prompt.

### Tasks

1. Use the same model setup from Component – I
2. Prepare the recipe instruction dataset and tokenize it
3. Fine-tune the model on recipe data
4. Generate recipe instructions using dish name prompts
5. Compare baseline vs. fine-tuned recipe outputs

### Dataset

Use the following recipe instruction corpus:

```
to make butter chicken start by marinating chicken pieces in yogurt with turmeric chili powder and garam masala for one hour.
...
```

### Code

#### Step 1: Reload Fresh Model

```
tokenizer2 = GPT2Tokenizer.from_pretrained('gpt2')
model2 = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer2.pad_token = tokenizer2.eos_token
model2.config.pad_token_id = tokenizer2.eos_token_id
```

#### Step 2: Generate Baseline Recipes

```
recipe_prompts = [
    'To make butter chicken',
    'For pasta carbonara',
    'To prepare a chocolate cake',
]
```

#### Step 3: Fine-Tune

```
# same pipeline as Component I
```

#### Step 4: Generate and Compare

```
eval2 = trainer2.evaluate()
print(f'Perplexity: {math.exp(eval2["eval_loss"]):.2f}')
```

### Expected Output

1. Baseline generates unrelated text
2. Fine-tuned model generates structured cooking instructions
3. Recipes follow logical cooking steps

---

## Models (Optional Alternatives)

DistilGPT-2, GPT-Neo, GPT-J, Phi-2, TinyLlama, LLaMA, Gemma, Qwen, BLOOM, OPT
