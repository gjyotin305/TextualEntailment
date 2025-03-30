import pandas as pd
from tqdm import tqdm
from nltk.tree import Tree
from unsloth import to_sharegpt, standardize_sharegpt, apply_chat_template
from datasets import load_dataset, Dataset, DatasetDict
import os

def extract_sentence(parse_tree: str) -> str:
    tree = Tree.fromstring(parse_tree)

    return ' '.join(tree.leaves())

def formatting_prompts_func(example):
    output_texts = []
    
    for i in range(len(example)):
        premise = example['Premise'][i]
        hypothesis = example['Hypothesis'][i]
        relation = example['Relation'][i]
    
        if len(premise) >= 2:
            text = f"### Premise: {premise}\n ### Hypothesis: {hypothesis}\n ### Relation: {relation}"
        else:
            text = f"### Premise: {premise}\n ### Hypothesis: {hypothesis}\n ### Relation: {relation}"

        output_texts.append(text)
    
    return output_texts

def convert_tsv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(f'{file_path}', sep='\t')
    df_real = pd.DataFrame()
    
    df_real['Premise'] = df['Sent1_parse']
    df_real['Hypothesis'] = df['Sent2_parse']
    df_real['Relation'] = df['Label']

    df_real['Premise'] = df_real['Premise'].map(
        lambda x : extract_sentence(x)
    )
    df_real['Hypothesis'] = df_real['Hypothesis'].map(
        lambda x : extract_sentence(x)
    )

    df_real.to_csv(f'{file_path}_modified.csv', index=False)

    return df_real

def convert_to_instruction_format(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(f'{file_path}')
    print(df.columns)
    df_real = pd.DataFrame()

    df_real['prompt'] = df.apply(lambda row: f"### Premise: {row['Premise']}\n### Hypothesis: {row['Hypothesis']}", axis=1)
    df_real['completion'] = df.apply(lambda row: f"### Relation: {row['Relation']}", axis=1)

    df_real.to_csv(f'{file_path}_modified.csv', index=False)

    return df_real

def make_dataset(folder_path: str, tokenizer) -> DatasetDict:
    valid_files = ["train.csv", "dev.csv", "test.csv"]

    data_files = {
        "train": f"{folder_path}{valid_files[0]}",
        "dev": f"{folder_path}{valid_files[1]}",
        "test": f"{folder_path}{valid_files[2]}"
    }

    dataset = load_dataset("csv", data_files=data_files)

    dataset_train = to_sharegpt(
        dataset=dataset['train'],
        merged_prompt="### Premise: {Premise}\n ### Hypothesis {Hypothesis}",
        output_column_name="Relation"
    )
    dataset_train = standardize_sharegpt(dataset_train)

    dataset_test = to_sharegpt(
        dataset=dataset['test'],
        merged_prompt="### Premise: {Premise}\n ### Hypothesis {Hypothesis}",
        output_column_name="Relation"
    )
    dataset_test = standardize_sharegpt(dataset_test)

    dataset_dev = to_sharegpt(
        dataset=dataset['dev'],
        merged_prompt="### Premise: {Premise}\n### Hypothesis: {Hypothesis}",
        output_column_name="Relation"
    )
    dataset_dev = standardize_sharegpt(dataset_dev)

    chat_template = """You are to perform textual entailment.

    ### Instruction:
    {INPUT}

    ### Response
    {OUTPUT}
    """

    dataset_train = apply_chat_template(
        dataset_train,
        tokenizer=tokenizer,
        chat_template=chat_template
    )

    dataset_test = apply_chat_template(
        dataset_test,
        tokenizer=tokenizer,
        chat_template=chat_template
    )
    
    dataset_dev = apply_chat_template(
        dataset_dev,
        tokenizer=tokenizer,
        chat_template=chat_template
    )

    final_dataset = DatasetDict({
        "train": dataset_train,
        "dev": dataset_dev,
        "test": dataset_test
    })

    return final_dataset

base_path = "/scratch/data/asif_rs/nli/data/inference_dataset/"

data_files = {
    "train": f"{base_path}train.csv",
    "dev": f'{base_path}dev.csv',
    "test": f'{base_path}test.csv'
}
dataset = load_dataset('csv', data_files=data_files, split="train")

dataset = to_sharegpt(
    dataset,
    merged_prompt="### Premise: {Premise}\n ### Hypothesis {Hypothesis}",
    output_column_name="Relation"
)
dataset = standardize_sharegpt(dataset)

print(dataset['conversations'][0])

chat_template = """
You are to perform textual entailment.

### Instruction:
{INPUT}

### Response
{OUTPUT}
"""

# train_dataset = dataset.map(formatting_prompts_func, batched=True)

# print(train_dataset)

# print(len(dataset))
# print(dataset)

