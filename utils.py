import pandas as pd
from tqdm import tqdm
from nltk.tree import Tree
import os

def extract_sentence(parse_tree: str) -> str:
    tree = Tree.fromstring(parse_tree)

    return ' '.join(tree.leaves())

def convert_tsv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(f'{file_path}', sep='\t')
    df_real = pd.DataFrame()
    
    df_real['Premise'] = df['Sent1_parse']
    df_real['Hypothesis'] = df['Sent2_parse']
    df_real['Relation'] = df['Label']

    df_real['Premise'] = df_real['Premise'].map(lambda x : extract_sentence(x))
    df_real['Hypothesis'] = df_real['Hypothesis'].map(lambda x : extract_sentence(x))

    df_real.to_csv(f'{file_path}_modified', index=False)

    return df_real


if __name__ == "__main__":
    base_path = "./data/inference_dataset/"
    list_dir = os.listdir(base_path)

    for x in tqdm(list_dir):
        if x.split('.')[-1] == "tsv":
            convert_tsv(f'{base_path}{x}')