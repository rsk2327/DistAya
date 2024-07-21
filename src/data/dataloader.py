import torch
from datasets import load_dataset
from torch.utils.data.dataset import Dataset

class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)

# Loading the data and tokenizing it
def process_data(samples, tokenizer, seq_len, field_name):
    test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len

    for i in range(nsamples):
        batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
        test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)
    return IndexDataset(tensors=test_ids_batch)

def get_loaders(tokenizer, seq_len=512, batch_size = 4, max_samples=256):
    test_data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    test_data = test_data.shuffle(seed=42)
    test_data = test_data.select(range(max_samples)) # select a small subset just for testing
    test_dataset = process_data(test_data, tokenizer, seq_len, 'text')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader
