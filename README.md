# ☢️ Watermarking Makes Language Models Radioactive ☢️

<p align="center">
    <img src="images/radioactive.jpg" width="300" height="auto">
</p>



We are excited to share that our work, which is now open-sourced, was accepted as a **spotlight at NeurIPS 2024**. 
This repository contains the code and additional resources related to our paper. 
For more detailed information, please refer to:


[[arxiv](https://arxiv.org/abs/2402.14904)] [[Webpage](https://ai.meta.com/research/publications/watermarking-makes-language-models-radioactive/)]

## Code

The code is adapted from the repository: https://github.com/facebookresearch/three_bricks which is licensed under the Creative Commons Attribution-NonCommercial 4.0 International Public License.


## Requirements

We use cuda 11.7. Please adapt the following to get the appropriate pytorch version.

```cmd
conda create -n "radioactive_watermark" python=3.8
conda activate radioactive_watermark
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

In the following, "<your_model_path>" corresponds the path of the model you want to test radioactivity on. 
You can for instance put the path to a Llama model. 
The seed is used when generating text in combination with the hashing key: two different seeds five different watermarking schemes.

## Data and models

- A subset of watermarked instruction/answer pairs with (Kirchenbauer et al.), watermark window size 2, $$\delta$$=3 and $$\gamma$$ = 0.25, seed = 3 can be found in `data/maryland_ngram2_seed3.jsonl`. We also provide a corresponding filter computed by saving the k-tuples and their frequencies in `data/maryland_ngram2_seed3_filter.pkl`.
- A similar dataset with a format compatible with the "reading mode" is available in `data/maryland_ngram4_seed0.jsonl`, but this time for a watermark window size 4 and seed 0.
- We store radioactive outputs of a model trained on 5% of watermarked data in `output_closed_supervised_0p05/results.jsonl` (window size 2, $$\delta$$=3 and $$\gamma$$ = 0.25, seed = 0), as well as some of the training data `data/used_maryland_ngram2_seed0.jsonl` and a filter computed from similar watermarked data in `data/used_maryland_ngram2_seed0_filter.pkl`.


## Usage - Closed model setting

</details>

For example, the following command analyses the radioactive outputs in `output_closed_supervised_0p05/results.jsonl` by concatenating the results and applying the deduplication proposed in section 4.
It corresponds the closed-model/supervised setting  with 5% of watermarked data in Figure 5.

Note that your model (in <your_model_path>) is not used here; the scripts notices the presence of outputs in `output_closed_supervised_0p05/`, so it does not generate any answers and just score the ones that are already present. 

```cmd
python main_watermark.py \
    --model_name <your_model_path> \
    --prompt_path "data/used_maryland_ngram2_seed0.jsonl" \
    --method none --method_detect maryland \
    --ngram 2 --scoring_method v2 \
    --nsamples 10000 --batch_size 16 \
    --output_dir output_closed_supervised_0p05/ \
    --filter_path "data/used_maryland_ngram2_seed0_filter.pkl" 
```


### Output

The previous script generates `results_chunked.jsonl`, which contains the following important fields, using the example of Kirchenbauer:

| Field | Description |
| --- | --- |
| `score` | Proportion of green list tokens until that point |
| `num_token` | Number of analyzed tokens in the text |
| `num_scored` | Number of scored tokens in the text |
| `pvalue` | p-value of the detection test |

The resulting file should look similar to `output_closed_supervised_0p05/result_chunked_expected.jsonl`
The final result should be close to 1e-30.

Running the following command will this time generate outputs from Llama-2-7b-chat-hf from the watermarked prompts, leading to a file `output_closed/results.jsonl` similar to `output_closed/results_expected.jsonl` and compute the radioactivity detection test to produce a result `result_chunked.jsonl` similar to `result_chunked_expected.jsonl`. 
This time, as the model was not trained on watermarked data, the resulting p-value should be random. 
Without deduplication (no_dedup = 1), it will appear falsely radioactive because the prompts are watermarked.
Results should be similar with or without the corresponding filter `data/maryland_ngram2_seed3_filter.pkl`.


```cmd
python main_watermark.py \
    --model_name <your_model_path> \
    --prompt_path "data/maryland_ngram2_seed3.jsonl" \
    --method none --method_detect maryland \
    --ngram 2 --scoring_method v2 --seed 3 \
    --nsamples 1000 --batch_size 16 \
    --output_dir output_closed/ \
    --filter_path "maryland_ngram2_seed3_filter.pkl" 
```



## Open model setting

The following command will run the reading mode with deduplication on "data/maryland_ngram4.jsonl"

```cmd
python main_reed_wm.py \
    --model_name  <your_model_path>\
    --dataset_path2 "data/reading_maryland_ngram4_seed0.jsonl" \
    --method_detect maryland \
    --nsamples 1000 \
    --batch_size 16 \
    --output_dir output_open/ \
    --ngram 4
```

If the model used is not radioactive, it should lead to a random p-value. 

## Compute filter

To compute a filter, follow the instructions of `create_filter.ipynb`

## Citation
If you find our work useful in your research, please consider citing:

```
@article{sander2024watermarking,
title={Watermarking Makes Language Models Radioactive},
author={Sander, Tom and Fernandez, Pierre and Durmus, Alain and Douze, Matthijs and Furon, Teddy},
journal={arXiv preprint arXiv:2402.14904},
year={2024}
}
```