from unsloth import FastLanguageModel
import re


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_te",
)

def extract_response(text):
    match = re.search(r"### Response\n(.*?)\n<\|eot_id\|>", text, re.DOTALL)
    return match.group(1).strip() if match else None    

def evaluate_single_pair(
    premise: str, 
    hypothesis: str, 
    relation: str, 
    model: FastLanguageModel,
    tokenizer
):
    FastLanguageModel.for_inference(model)
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

    result = tokenizer.decode(response[0])
    result = extract_response(result)

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