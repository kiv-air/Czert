# CZERT
This repository keeps trained models for the paper [Czert – Czech BERT-like Model for Language Representation
](https://arxiv.org/abs/2103.13031)
For more information, see the paper


## Available Models
You can download **MLM & NSP only** pretrained models
[CZERT-A](https://air.kiv.zcu.cz/public/CZERT-A-czert-albert-base-uncased.zip)
[CZERT-B](https://air.kiv.zcu.cz/public/CZERT-B-czert-bert-base-cased.zip)

or choose from one of **Finetuned Models**
| | Models  |
| - | - |
| Sentiment Classification<br> (Facebook or CSFD)                                                                                                                           | [CZERT-A-sentiment-FB](https://air.kiv.zcu.cz/public/CZERT-A_fb.zip) <br> [CZERT-B-sentiment-FB](https://air.kiv.zcu.cz/public/CZERT-B_fb.zip) <br> [CZERT-A-sentiment-CSFD](https://air.kiv.zcu.cz/public/CZERT-A_csfd.zip)  <br>   [CZERT-B-sentiment-CSFD](https://air.kiv.zcu.cz/public/CZERT-B_csfd.zip) | Semantic Text Similarity <br> (Czech News Agency)                                                                                                                        | [CZERT-A-sts-CNA](https://air.kiv.zcu.cz/public/CZERT-A-sts-CNA.zip) <br> [CZERT-B-sts-CNA](https://air.kiv.zcu.cz/public/CZERT-B-sts-CNA.zip)                                                                                                                                               
| Named Entity Recognition                                                                                                                                                 | [CZERT-A-ner-CNEC](https://air.kiv.zcu.cz/public/CZERT-A-ner-CNEC.zip) <br>  [CZERT-B-ner-CNEC](https://air.kiv.zcu.cz/public/CZERT-B-ner-CNEC.zip) <br>[PAV-ner-CNEC](https://air.kiv.zcu.cz/public/PAV-ner-CNEC.zip) <br> [CZERT-A-ner-BSNLP](https://air.kiv.zcu.cz/public/CZERT-A-ner-BSNLP.zip)<br>[CZERT-B-ner-BSNLP](https://air.kiv.zcu.cz/public/CZERT-B-ner-BSNLP.zip) <br>[PAV-ner-BSNLP](https://air.kiv.zcu.cz/public/PAV-ner-BSNLP.zip) |
| Morphological Tagging<br> | [CZERT-A-morphtag-126k](https://air.kiv.zcu.cz/public/CZERT-A-morphtag-126k.zip)<br>[CZERT-B-morphtag-126k](https://air.kiv.zcu.cz/public/CZERT-B-morphtag-126k.zip)                                                                                                                                                                                                                                                                                  |
| Semantic Role Labelling                                                                                                                                                  |[CZERT-A-srl](https://air.kiv.zcu.cz/public/CZERT-A-srl.zip)<br>                                              [CZERT-B-srl](https://air.kiv.zcu.cz/public/CZERT-B-srl.zip)                                                                                                                                                                                                                                                                                                    |





## How to Use CZERT?

### Sentence Level Tasks
We evaluate our model on two sentence level tasks:
* Sentiment Classification,
* Semantic Text Similarity.



<!--     tokenizer = BertTokenizerFast.from_pretrained(CZERT_MODEL_PATH, strip_accents=False)  
	model = TFAlbertForSequenceClassification.from_pretrained(CZERT_MODEL_PATH, num_labels=1)
    
or
    
    self.tokenizer = BertTokenizerFast.from_pretrained(CZERT_MODEL_PATH, strip_accents=False)
    self.model_encoder = AutoModelForSequenceClassification.from_pretrained(CZERT_MODEL_PATH, from_tf=True)
     -->
	
### Document Level Tasks
We evaluate our model on one document level task
* Multi-label Document Classification.

### Token Level Tasks
We evaluate our model on three token level tasks:
* Named Entity Recognition,
* Morphological Tagging,
* Semantic Role Labelling. 


## Downstream Tasks Fine-tuning Results

### Sentiment Classification
|      |          mBERT           |        SlavicBERT        |         ALBERT-r         |         Czert-A         |             Czert-B              |
|:----:|:------------------------:|:------------------------:|:------------------------:|:-----------------------:|:--------------------------------:|
|  FB  | 71.72 ± 0.91   | 73.87 ± 0.50  | 59.50 ± 0.47  | 72.47 ± 0.72  | **76.55** ± **0.14** |
| CSFD | 82.80 ± 0.14   | 82.51 ± 0.14  | 75.40 ± 0.18  | 79.58 ± 0.46  | **84.79** ± **0.26** |

Average F1 results for the Sentiment Classification task. For more information, see [the paper](https://arxiv.org/abs/2103.13031). 
                 

### Semantic Text Similarity

|              |   **mBERT**    |   **Pavlov**   | **Albert-random** |  **Czert-A**   |      **Czert-B**       |
|:-------------|:--------------:|:--------------:|:-----------------:|:--------------:|:----------------------:|
| STA-CNA      | 83.335 ± 0.063 | 83.593 ± 0.050 |  43.184 ± 0.125   | 82.942 ± 0.106 | **84.345** ± **0.028** |
| STS-SVOB-img | 79.367 ± 0.486 | 79.900 ± 0.810 |  15.739 ± 2.992   | 79.444 ± 0.338 | **83.744** ± **0.395** |
| STS-SVOB-hl  | 78.833 ± 0.296 | 76.996 ± 0.305 |  33.949 ± 1.807   | 75.089 ± 0.806 |     **79.827 ± 0.469**     |

Comparison of Pearson correlation achieved using pre-trained CZERT-A, CZERT-B, mBERT, Pavlov and randomly initialised Albert on semantic text similarity. For more information see [the paper](https://arxiv.org/abs/2103.13031).




### Multi-label Document Classification
|       |    mBERT     |  SlavicBERT  |   ALBERT-r   |   Czert-A    |      Czert-B        |
|:-----:|:------------:|:------------:|:------------:|:------------:|:-------------------:|
| AUROC | 97.62 ± 0.08 | 97.80 ± 0.06 | 94.35 ± 0.13 | 97.49 ± 0.07 | **98.00** ± **0.04** |
|  F1   | 83.04 ± 0.16 | 84.08 ± 0.14 | 72.44 ± 0.22 | 82.27 ± 0.17 | **85.06** ± **0.11** |

Comparison of F1 and AUROC score achieved using pre-trained CZERT-A, CZERT-B, mBERT, Pavlov and randomly initialised Albert on multi-label document classification. For more information see [the paper](https://arxiv.org/abs/2103.13031).

### Morphological Tagging
|                        | mBERT          | Pavlov         | Albert-random  | Czert-A        | Czert-B        |
|:-----------------------|:---------------|:---------------|:---------------|:---------------|:---------------|
| Universal Dependencies | 99.176 ± 0.006 | 99.211 ± 0.008 | 96.590 ± 0.096 | 98.713 ± 0.008 | **99.273 ± 0.006** |

Comparison of F1 score achieved using pre-trained CZERT-A, CZERT-B, mBERT, Pavlov and randomly initialised Albert on morphological tagging task. For more information see [the paper](https://arxiv.org/abs/2103.13031).
### Semantic Role Labelling

<div id="tab:SRL">

|        |   mBERT    |   Pavlov   | Albert-random |  Czert-A   |  Czert-B   | dep-based | gold-dep |
|:------:|:----------:|:----------:|:-------------:|:----------:|:----------:|:---------:|:--------:|
|  span  | 78.547 ± 0.110 | **79.333 ± 0.080** |  51.365 ± 0.423   | 72.254 ± 0.172 | **79.112 ± 0.141** |    \-     |    \-    |
| syntax | 90.226 ± 0.224 | **90.492 ± 0.040** |  80.747 ± 0.131   | 80.319 ± 0.054 | **90.516 ± 0.047** |   85.19   |  89.52   |

SRL results – dep columns are evaluate with labelled F1 from CoNLL 2009 evaluation script, other columns are evaluated with span F1 score same as it was used for NER evaluation. For more information see [the paper](https://arxiv.org/abs/2103.13031).

</div>


### Named Entity Recognition
|            | mBERT          | Pavlov         | Albert-random  | Czert-A        | Czert-B        |
|:-----------|:---------------|:---------------|:---------------|:---------------|:---------------|
| CNEC       | **86.225 ± 0.208** | **86.565 ± 0.198** | 34.635 ± 0.343 | 72.945 ± 0.227 | 81.632 ± 0.165 |
| BSNLP 2019 | 84.006 ± 1.248 | **86.699 ± 0.370** | 19.773 ± 0.938 | 48.859 ± 0.605 | 80.320 ± 1.090 |

Comparison of f1 score achieved using pre-trained CZERT-A, CZERT-B, mBERT, Pavlov and randomly initialised Albert on named entity recognition task. For more information see [the paper](https://arxiv.org/abs/2103.13031).


## How should I cite CZERT? 
For now, please cite [the Arxiv paper](https://arxiv.org/abs/2103.13031):
```
@article{sido2021czert,
      title={Czert -- Czech BERT-like Model for Language Representation}, 
      author={Jakub Sido and Ondřej Pražák and Pavel Přibáň and Jan Pašek and Michal Seják and Miloslav Konopík},
      year={2021},
      eprint={2103.13031},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      journal={arXiv preprint arXiv:2103.13031},
}
```
