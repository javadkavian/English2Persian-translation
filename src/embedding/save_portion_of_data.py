from datasets import load_from_disk
import numpy as np
import pandas as pd

NUMBER_OF_SAMPLES = 200000

if __name__ == "__main__":
    dataset = load_from_disk('././dataset/')
    
    
    targets_df = pd.DataFrame([item[0] for item in dataset['train']['targets']], columns=['targets'])
    source_df = pd.DataFrame(dataset['train']['source'], columns=['source'])
    
    
    idxs = np.random.choice(len(targets_df), NUMBER_OF_SAMPLES, replace=False)
    
    
    sampled_targets = targets_df.iloc[idxs]['targets']
    sampled_sources = source_df.iloc[idxs]['source']
    
    
    df = pd.DataFrame({'persian': sampled_targets, 'english': sampled_sources})
    df.to_csv('././dataset/shortened_dataset.csv', index=False)
