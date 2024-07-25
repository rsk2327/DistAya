# Pruning
## Layer Sensitivity Scores
The first step is to compute the sensitivity of each layer to pruning. This gives us an idea of the importance of each layer.

### Perplexity as a Sensitivity Score
The approach is to examine the perplexity induced when removing a given layer.

```bash
python -m DistAya.src.pruning.perplexity_sensivity \
            --model CohereForAI/aya-23-8B \
            --batch_size 8 \
            --output_folder sensitivities \
            --subset 128
```

This will produce a CSV representing the sensitivity of each layer to pruning. This sensitivity score is just the perplexity of the model when this layer is dropped.


### Input/output similarity as a Sensivity Score
See the [ShortGPT Paper](https://arxiv.org/abs/2403.03853)

## Compression
...

# Distillation
...
