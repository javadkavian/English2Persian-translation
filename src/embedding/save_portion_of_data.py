from datasets import load_from_disk
import numpy as np
import pandas as pd

NUMBER_OF_SAMPLES = 200000

if __name__ == "__main__":
    dataset = load_from_disk('././dataset/')
    
    # Convert the nested data to DataFrame and normalize it
    targets_df = pd.DataFrame([item[0] for item in dataset['train']['targets']], columns=['targets'])
    source_df = pd.DataFrame(dataset['train']['source'], columns=['source'])
    
    # Randomly sample indices
    idxs = np.random.choice(len(targets_df), NUMBER_OF_SAMPLES, replace=False)
    
    # Use .iloc to select the samples
    sampled_targets = targets_df.iloc[idxs]['targets']
    sampled_sources = source_df.iloc[idxs]['source']
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame({'persian': sampled_targets, 'english': sampled_sources})
    df.to_csv('././dataset/shortened_dataset.csv', index=False)
