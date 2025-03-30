import re 
from tqdm import tqdm
import pandas as pd
import json
import numpy

df = pd.read_csv(
    "./Task_B_Dataset.csv"
)

def consolidate_json(file_path: str, pattern_type):
    result = {
        "pattern": f"{pattern_type}",
        "sents": []
    }

    with open(f'{file_path}', 'r') as f:
        data = json.load(f)
        f.close()
    
    for item in data:
        if "sents" in item and isinstance(item['sents'], list):
            result['sents'].extend(item['sents'])
    
    with open(f'{file_path}', 'w') as f:
        data = json.dump(result, f, indent=4, cls=NpEncoder)
        f.close()

    return result

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def extract_pattern1(filename, para_id, sent_id, raw_text):
    """
    Extract phrases matching Pattern 1 (in tag followed by nn tags) using regex
    """
    # Initialize result structure
    result = {
        "pattern": "pattern 1",
        "sents": []
    }
    
    # Extract plain text (words without tags)
    plain_text = " ".join([token.split('/')[0] for token in raw_text.split() if '/' in token])
    
    # Pattern: word/in followed by one or more word/nn
    pattern = r'(\S+/in)(?:\s+(\S+/nn))+' 
    matches = re.finditer(pattern, raw_text)
    
    phrases = []
    for match in matches:
        matched_text = match.group(0)
        
        # Extract all tokens in the match
        tokens = matched_text.split()
        
        # Extract words without tags
        words = [token.split('/')[0] for token in tokens]
        phrase_text = " ".join(words)
        
        # Determine phrase type
        nn_count = len(tokens) - 1  # All tokens except the first 'in' token
        phrase_type = f"in {' '.join(['nn'] * nn_count)}"
        
        # Find begin and end positions in plain text
        begin = plain_text.find(phrase_text)
        end = begin + len(phrase_text)
        
        phrases.append({
            "begin": int(begin),
            "end": int(end),
            "text": phrase_text,
            "phrase_type": phrase_type
        })
    
    # Add sentence entry if phrases were found
    if phrases:
        result["sents"].append({
            "filename": filename,
            "para_id": para_id,
            "sent_id": sent_id,
            "sent_text": plain_text,
            "phrases": phrases
        })
    
    return result

def extract_pattern2(filename, para_id, sent_id, raw_text):
    """
    Extract phrases matching Pattern 2 (jj tag followed by nn tags) using regex
    """
    # Initialize result structure
    result = {
        "pattern": "pattern 2",
        "sents": []
    }
    
    # Extract plain text (words without tags)
    plain_text = " ".join([token.split('/')[0] for token in raw_text.split() if '/' in token])
    
    # Pattern: word/in followed by one or more word/nn
    pattern = r'(\S+/jj)(?:\s+(\S+/nn))+' 
    matches = re.finditer(pattern, raw_text)
    
    phrases = []
    for match in matches:
        matched_text = match.group(0)
        
        # Extract all tokens in the match
        tokens = matched_text.split()
        
        # Extract words without tags
        words = [token.split('/')[0] for token in tokens]
        phrase_text = " ".join(words)
        
        # Determine phrase type
        nn_count = len(tokens) - 1  # All tokens except the first 'in' token
        phrase_type = f"jj {' '.join(['nn'] * nn_count)}"
        
        # Find begin and end positions in plain text
        begin = plain_text.find(phrase_text)
        end = begin + len(phrase_text)
        
        phrases.append({
            "begin": int(begin),
            "end": int(end),
            "text": phrase_text,
            "phrase_type": phrase_type
        })
    
    # Add sentence entry if phrases were found
    if phrases:
        result["sents"].append({
            "filename": filename,
            "para_id": para_id,
            "sent_id": sent_id,
            "sent_text": plain_text,
            "phrases": phrases
        })
    
    return result

if __name__ == "__main__":
    # print(len(df))
    
    # df.dropna()
    responses = []
    
    for i in tqdm(range(len(df))):
        res = extract_pattern2(
            filename=df['filename'][i],
            para_id=df['para_id'][i],
            sent_id=df['sent_id'][i],
            raw_text=df['raw_text'][i]
        )
        responses.append(res)

    with open(
        './pattern_2.json', 'w'
    ) as f:
        json.dump(responses, f, indent=4, cls=NpEncoder)
        f.close()  

    responses = []
    
    for i in tqdm(range(len(df))):
        res = extract_pattern1(
            filename=df['filename'][i],
            para_id=df['para_id'][i],
            sent_id=df['sent_id'][i],
            raw_text=df['raw_text'][i]
        )
        responses.append(res)

    with open(
        './pattern_1.json', 'w'
    ) as f:
        json.dump(responses, f, indent=4, cls=NpEncoder)
        f.close()  

    consolidate_json(
        file_path="./pattern_1.json",
        pattern_type="pattern 1"
    )
    consolidate_json(
        file_path="./pattern_2.json",
        pattern_type="pattern 2"
    )

    # res = extract_pattern2(
    #     filename=df['filename'][0],
    #     para_id=df['para_id'][0],
    #     sent_id=df['sent_id'][0],
    #     raw_text="A/at large/jj piece/nn of/in engine/nn cowling/nn vanished/vbd ./."
    # )
    # print(res)

# print(df)
# print(df.columns)