# 生成式方面情感三元组抽取

Seq2seq for Aspect Sentiment Triplet Extraction

### 方面情感三元组抽取

方面情感三元组(Aspect Sentiment Triplet Extraction, ASTE)任务旨在抽取用户评论中表达观点的方面情感三元组。一个三元组包含以下三个部分：

- aspect term: 情感所针对的目标对象，一般是被评价实体的某个方面项，常被称作方面词、属性词
- opinion term: 具体表达情感的词或短语，常被称作情感词
- sentiment polarity: 用户针对aspect term所表达的情感倾向，类别空间为`{POS, NEG, NEU}`

<div align="center"> <img src="https://user-images.githubusercontent.com/9134454/199022562-2cca1c06-b91e-4e4b-8bf0-20273a16821e.png" alt="ASTE" width="50%" /></div>

### 生成式方法

生成式方法，(1)首先将ASTE任务转化为一个seq2seq任务，(2) 然后使用一个seq2seq的模型(如t5)来建模。

	输入: average to good thai food but incredibly slow and rude delivery
	输出: thai food | average to good | positive ; delivery | incredibly slow | negative ; delivery | rude | negative

### 结果

seed=42

|           | 14res  | 14lap  | 15res  | 16res  |
| --------- | ------ | ------ | ------ | ------ |
| precision | 0.7490 | 0.6727 | 0.6326 | 0.7362 |
| recall    | 0.7364 | 0.6188 | 0.6247 | 0.7763 |
| f1-score  | 0.7426 | 0.6446 | 0.6286 | 0.7557 |

### 运行代码

**Requirements**

	transformers==4.26.1
	pytorch==1.10.1
	pytorch-lightning==1.9.3
	spacy==3.5.0

**预处理数据**

	python t5_convert.py --raw_data_dir data/raw --output_data_dir data/t5 --dataset 14res

**训练模型**

	chmod +x bash/*
	bash/train_extractor.sh -d 14res -c 3 -o {YOUR_OUTPUT_DIR}

### 参考

- 数据来自 https://github.com/xuuuluuu/SemEval-Triplet-data
- Zhang W, Li X, Deng Y, et al. Towards generative aspect-based sentiment analysis[C]//Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers). 2021: 504-510.


