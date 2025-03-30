from unsloth import FastLanguageModel
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report
import re

label2index = {
    "entailment": 0,
    "contradiction": 1,
    "neutral": 2
}

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_te_2",
)

def extract_responses(text_list):
    responses = []
    pattern = re.compile(r"### Response\n(.*?)\n<\|eot_id\|>", re.DOTALL)
    
    for text in text_list:
        match = pattern.search(text)
        responses.append(match.group(1).strip() if match else None)
    
    return responses 

def evaluate_dataset_pair(
    file_path: str,
    batch_size: int,
    model: FastLanguageModel,
    tokenizer
):
    df = pd.read_csv(file_path)
    # df.drop(df[])
    print(df.columns)

    df['prompt'] = df.apply(
        lambda x: f"### Premise: {x['Premise']}\n### Hypothesis: {x['Hypothesis']}", axis=1
    )
    
    FastLanguageModel.for_inference(model)
    messages = [
        [{"role": "user", "content": f"{prompt}"}] for prompt in df['prompt']
    ]
    responses = []

    for index in tqdm(range(0, len(messages), batch_size)):
        if (index+batch_size) < len(messages):
            message = messages[index: index + batch_size]
        else:
            message = messages[index:]

        input_ids = tokenizer.apply_chat_template(
            message,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        # print(input_ids.shape)

        response = model.generate(
            input_ids, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id
        )

        # print(response.shape)
        
        result = tokenizer.batch_decode(response)
        
        result = extract_responses(result)

        # print(result)

        responses.extend(result)
    
    print(f"{len(responses)}| {len(df['Relation'])}")

    assert len(responses) == len(df['Relation'])

    ground_truth = [label2index[y] for y in tqdm(df['Relation'])]
    pred = [label2index[response] for response in tqdm(responses)]

    print(precision_recall_fscore_support(ground_truth, pred))
    print(classification_report(ground_truth, pred))


def evaluate_single_pair(
    premise: str, 
    hypothesis: str, 
    relation: str, 
    model: FastLanguageModel,
    tokenizer
):
    FastLanguageModel.for_inference(model)
    responses = []

    messages = [
        {"role": "user", "content": f"### Premise: {premise}\n### Hypothesis: {hypothesis}"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    response = model.generate(
        input_ids, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id
    )

    result = tokenizer.batch_decode(response)
    result = extract_responses(result)

    print(f"Ground Truth {relation} | Predicted {result}")

    return result


if __name__ == "__main__":
    evaluate_single_pair(
        premise="A man in a black leather jacket and a book in his hand speaks in a classroom .",
        hypothesis="A man is speaking in a classroom .",
        relation="entailment",
        model=model,
        tokenizer=tokenizer
    )
    evaluate_dataset_pair(
        file_path="/scratch/data/asif_rs/nli/data/inference_dataset/test.csv",
        batch_size=1,
        model=model,
        tokenizer=tokenizer
    )