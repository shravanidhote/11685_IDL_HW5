## 11-685: Introduction to Deep Learning

## Homework 5: Build Your LLM

#### Authors: Shreya Kale & Shravani Dhote

Our project focuses on the development and evaluation of a Large Language Model (LLM) fine-tuned for the summarization and question-answering tasks. The LLM employs a decoder-only transformer architecture, initially pretrained on a subset of the OpenWebText dataset. Task-specific fine-tuning is conducted using the CNN Daily Mail and SQuAD datasets for summarization and question-answering tasks respectively. Quantitative performance evaluation utilizes BLEU for summarization and ROUGE for question answering. This research contributes useful insights into the design and training practices for custom LLMs tailored to specific NLP tasks while considering constraints related to computational resources. The findings aim to guide researchers and practitioners in leveraging LLMs effectively within limited computing environments.

## Directory structure

```
   ./Jupyter_Notebooks
    |
    |--- Get_dataset.ipynb: -- Notebook for downloading, processing and combining the CNN and SQUAD data
    |--- HW5_pretraining.ipynb -- Notebook for pretraining the model
    |--- Hw5_finetuning.ipynb: -- Notebook for finetuning the model
    |--- HW5_testing.ipynb -- Notebook for testing and obtaining predictions
```
```
   ./Model_Description
    |
    |--- our_model_summary.csv: -- Contains our model description
```



## Steps for Execution
1. We implement a transformer-based neural architecture composed solely of a decoder. The input is text, such as news articles and paragraphs, which the model encodes into continuous representations. We pretrain this model on a subset of the OpenWebText dataset to learn strong natural language foundations using the cells present in `HW5_pretraining.ipynb`. Keep it to train for a couple of days till one can see improvement in context comprehension and coherence.
2. Then Execute the `Get_dataset.ipynb` notebook to generate a shuffled combined data summarization and QA tasks. We fine-tune the model on two key datasets to enhance performance on question-answering and summarization tasks. The Stanford Question Answering Dataset (SQuAD) provides crowd-sourced questions and Wikipedia-based answers to train the model for question answering. Using these labelled question-passage-answer triplets tunes the model to select accurate span-based answers. Additionally, the CNN Daily Mail dataset compiles news articles paired with multi-sentence summaries. 
3. Next, we perform task-specific finetuning separately on the CNN Daily Mail and SQuAD datasets for summarization and question answering correspondingly. Fine-tuning leverages a pre-trained model to enhance performance on downstream tasks. We fine-tune using the model checkpoint from initial pretraining, reusing the training framework including the dataloader and language modeling loss. Fine-tuning on these data trains the model to condense texts and capture key information. By leveraging the complementary strengths of SQuAD and CNN, we adapt the pretrained foundations to excel at extraction-based question answering while also becoming proficient at text summarization. Reload the processed data in `Hw5_finetuning.ipynb` and train it for a few epochs till the training loss decreases further and the predictions start making sense.
4. Load the best model checkpoint from finetuning into `HW5_testing.ipynb` and execute the cells to get the predictions and the BLEU and ROUGE scores.
