## <p align=center>`Fine-tuned Longformer for Summarization of Machine Learning Articles`</p>
## Check the [Poster](https://github.com/Bakhitovd/led-base-7168-ml/blob/main/Bakhitov%20Summarization%20of%20Specific%20Domain%20Long%20Documents%20through%20Fine-Tuning%20Longformer.pdf) first
# Model
The [led-base-7168-ml](https://huggingface.co/bakhitovd/led-base-7168-ml) is a `Longformer`  model fine-tuned on a [ML_arxiv](https://huggingface.co/datasets/bakhitovd/ML_arxiv) dataset containing articles related to machine-learning topics for long-document summarization task.

The [led-base-7168-ml](https://huggingface.co/bakhitovd/led-base-7168-ml) is available on [Hugging Face](https://huggingface.co/bakhitovd/led-base-7168-ml). This is [`led-base-16384`](https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-encdec-base-16384.tar.gz) pretrained transformer model fine-tuned on the [ML_arxiv](https://huggingface.co/datasets/bakhitovd/ML_arxiv) dataset for summarization task. This model is able to effectively generate a coherent and consistent summary of a long article (up to 16384 tokens) related to machine learning topics.
To use, try something like the following:

```python
import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration
tokenizer = LEDTokenizer.from_pretrained("bakhitovd/led-base-7168-ml")
model = LEDForConditionalGeneration.from_pretrained("bakhitovd/led-base-7168-ml")

```
```python
article = "... long document ..."
inputs_dict = tokenizer.encode(article, padding="max_length", max_length=16384, return_tensors="pt", truncation=True)
input_ids = inputs_dict.input_ids.to("cuda")
attention_mask = inputs_dict.attention_mask.to("cuda")
global_attention_mask = torch.zeros_like(attention_mask)
global_attention_mask[:, 0] = 1
predicted_abstract_ids = model.generate(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, max_length=512)
summary = tokenizer.decode(predicted_abstract_ids, skip_special_tokens=True)
print(summary)

```
or you can use `summarization.ipynb` notebook. This code extracts the content of an online article, generates a summary of the article using a pretrained transformer model `led-base-7168-ml`, and then displays the summary in an HTML format in an IPython environment. The script uses BeautifulSoup for web scraping, requests for HTTP requests, and PyTorch along with the transformers library for summarization.

# Dataset

The [ML_arxiv](https://huggingface.co/datasets/bakhitovd/ML_arxiv) is a dataset of long structured documents obtained from the ArXiv OpenAccess repository. The `ML_arxiv` is a subset of the [scientific papers](https://github.com/armancohan/long-summarization) dataset established in ["A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents"](https://arxiv.org/abs/1804.05685) and widely used for summarization of long scientific documents.

'Machine learning' articles were extracted by clustering of embeddings of articles abstracts. To do that 2,077 machine learning-related articles were identified within the `scientific papers` dataset. Then all articles of the `scientific papers` dataset were clustered into 6 clusters and the closest by cosine similarity to 2,077 machine learning-related articles cluster was selected. 

It is not guaranteed that `ML_arxiv` contains only articles about machine learning, but the `ML_arxiv` contains 32,621 instances of the `scientific papers` dataset that are semantically, vocabulary-wise, structurally, and meaningfully closest to articles describing machine learning. 

This dataset could be used for training models for summarization of long texts of machine learning specific domain.

You can access the dataset in Huggingface datasets library:

https://huggingface.co/datasets/bakhitovd/ML_arxiv


# Evaluation
The performance of the fine-tuned model was evaluated using the ROUGE metrics, which measure the overlap between the ground truth summary and the generated summary in terms of unigrams (ROUGE-1), bigrams (ROUGE-2), and the longest contiguous common sequence (ROUGE-L).

![image](https://user-images.githubusercontent.com/55272111/236964399-e67f8134-a0d4-4e77-a1c2-439f0d967204.png)

