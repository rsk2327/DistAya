from ..data.dataloader import get_loaders
from ..utils import setmodule, getmodule
from  pathlib import Path
import re
from argparse import ArgumentParser
from pandas import DataFrame
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

class PruneLayer(torch.nn.Module):
    """"If pruned, the layer will do nothing other than returning its input."""
    def __init__(self, layer, is_last: bool, drop: bool=False):
        super().__init__()
        self.layer = layer
        self.drop = drop
        self.is_last = is_last

    @torch.no_grad()
    def forward(self, hidden_states, **kwargs):
        if self.drop:
            return (hidden_states,)
        return self.layer(hidden_states, **kwargs)

def get_layers(llm):
    for name, module in llm.named_modules():
        if re.search("layers\.\d+$", name):
            yield name, module

def set_pruned_layers(llm):
    for name, module in get_layers(llm):
        pruned_layer = PruneLayer(module, drop=False, is_last=False)
        setmodule(llm, name, pruned_layer)

def prune_layer(llm, target_module):
    module = getmodule(llm, target_module)
    module.drop = True

def unprune_layer(llm, target_module):
    module = getmodule(llm, target_module)
    module.drop = False

@torch.no_grad()
def llm_eval(model, test_lodaer, device):
    nlls = []
    for batch in tqdm(test_lodaer):
        batch = batch.to(device)
        output = model(batch, use_cache=False, output_attentions=False)
        lm_logits = output.logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()

def parse_args():
    parser = ArgumentParser(description="Computing the sensivity to pruning for each layer. The sensivity score is base on perplexity.")
    parser.add_argument("-m", "--model",
                        type=str,
                        help="The LLM.")
    parser.add_argument("-s", "--subset",
                        type=int,
                        default=512,
                        help="The number of examples to use for computing the sensivities score.")
    parser.add_argument("-b", "--batch_size",
                        type=int,
                        help="The batch size for running the evaluation.",
                        required=False,
                        default=64)
    parser.add_argument("-o", "--output_folder",
                        type=str,
                        help="Where to store the sensivity scores.")
    
    return parser.parse_args()
def main():
    args = parse_args()
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    llm = AutoModelForCausalLM.from_pretrained(args.model,
                                               torch_dtype=torch.bfloat16,
                                               trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=True)
    llm.cuda()
    test_loader = get_loaders(tokenizer=tokenizer, max_samples=args.subset, batch_size=args.batch_size)
    set_pruned_layers(llm)

    base_ppl = llm_eval(llm, test_loader, "cuda")
    layers = list(name for name, _ in get_layers(llm))
    results = []
    for name in tqdm(layers):
        layer_idx = name.split(".")[-1]
        prune_layer(llm, target_module=name)
        ppl = llm_eval(llm, test_loader, "cuda")
        unprune_layer(llm, target_module=name)
        print(f"pruned layer={layer_idx}, perplexity={ppl}, base perplexity={base_ppl}")
        results.append({
            "layer": name,
            "score": ppl,
        })
        df = DataFrame(results)
        df.to_csv(output_folder / f"perplexity_sensivities.csv")

if __name__ == "__main__":
    main()