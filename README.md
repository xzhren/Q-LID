# Q-LID

Implementation for the [CL] Computational Linguistics paper "Effective Approaches to Neural Query Language Identification".


### Requirements
- Python = 3.6 (or 2.7)
- TensorFlow = 1.12.0 (>= 1.4.0)
- pyyaml
- nltk


### Benchmark
- **QID-21:  in [corpus_langs104/test23fy21](https://github.com/xzhren/Q-LID/tree/main/corpus_langs104/test23fy21)**

The "**QID-21**" is collected from a real-world search engine -- [AliExpress](https://www.aliexpress.com/) who is an online international retail service. This benchmark consists of 21 languages and 21,440 samples. The average word count in each sample is 2.56, and the average number with respect to character is 15.53.


- **KB-21:  in [corpus_langs104/testkb21](https://github.com/xzhren/Q-LID/tree/main/corpus_langs104/testkb21)**

The "**KB-21**" is a publicly available test set from Kocmi and Bojar (2017)[1], using a subset of 21 languages. "KB-21" consists of 2,100 samples, the average amounts of words and characters in each sample are 4.47 and 34.90, respectively.


- **Explanation**

The file test.src records the original text, and the file test.trg records the language code of the text.

- **Language label and abbreviations**

English (en), Chinese (zh), Russian (ru), Portuguese (pt), Spanish (es), French (fr), German (de), Italian (it), Dutch (nl), Japanese (ja), Korean (ko), Arabic (ar), Thai (th), Hindi (hi), Hebrew (he),  Vietnamese (vi), Turkish (tr),  Polish (pl),  Indonesian (id), Malay (ms), and Ukrainian (uk).


### Model Instruction
- **use_word_script_embedding**: use word and script feature.
- **use_subword_embedding**: use sub-word (bpe) feature.
- **vocab_size**: the size of character feature vocab.
- **src_word_vocab_size**: the size of word or sub-word feature vocab.
- **class_num**: the number of support languages.

### Train & Evaluation
- Train
> python train_transformer.py --exp_name qlid_transformer_vbase_langs104 --corpus_dir corpus_langs104 --vocab_dir corpus_langs104 --vocab_size 10000 --class_num 104 --use_new_net True --use_word_script_embedding True --src_word_vocab_size 60000 --use_subword_embedding True

- Evaluation on QID-21 Testset
> python eval.py --exp_name qlid_transformer_vbase_langs104 --corpus_dir corpus_langs104/test23fy21 --vocab_dir corpus_langs104 --vocab_size 10000 --class_num 104 --use_new_net True --use_word_script_embedding True --src_word_vocab_size 60000 --use_subword_embedding True --postfix ".QID21" --eval23 True

- Evaluation on KB-21 Testset
> python eval.py --exp_name qlid_transformer_vbase_langs104 --corpus_dir corpus_langs104/testkb21 --vocab_dir corpus_langs104 --vocab_size 10000 --class_num 104 --use_new_net True --use_word_script_embedding True --src_word_vocab_size 60000 --use_subword_embedding True --postfix ".KB21" --eval23 True

- Evaluation on LID-104  Testset
> python eval.py --exp_name qlid_transformer_vbase_langs104 --corpus_dir corpus_langs104 --vocab_dir corpus_langs104 --vocab_size 10000 --class_num 104 --use_new_net True --use_word_script_embedding True --src_word_vocab_size 60000 --use_subword_embedding True --postfix ".LID104"

### Models
- **QID-104:  in [logs/best_models/export/](https://github.com/xzhren/Q-LID/tree/main/logs/best_models/export/)**
  - logs/best_models/export/label.txt
  - logs/best_models/export/saved_model.pb
  - logs/best_models/export/vocab.txt
  - logs/best_models/export/vocab_bpe.txt
  - logs/best_models/export/variables/variables.data-00000-of-00001
  - logs/best_models/export/variables/variables.index


### References
\[1\] Tom Kocmi and Ondrej Bojar. 2017. Lanidenn: Multilingual language identification on character window. CoRR, abs/1701.03338.

