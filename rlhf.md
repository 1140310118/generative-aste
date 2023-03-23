## seq2seq原本的性能

seed=42

|           | 14res  | 14lap  | 15res  | 16res  |
| --------- | ------ | ------ | ------ | ------ |
| precision | 0.7490 | 0.6727 | 0.6326 | 0.7362 |
| recall    | 0.7364 | 0.6188 | 0.6247 | 0.7763 |
| f1-score  | 0.7426 | 0.6446 | 0.6286 | 0.7557 |

## beam

	Its good to go there for drinks if you don't want to get drunk because you'll be lucky if you can get one drink an hour the service is so bad.
	0 drinks | good | positive ; service | bad | negative
	1 drink | good | positive ; service | bad | negative
	√ drinks | good | neutral ; service | bad | negative
	3 drink | good | neutral ; service | bad | negative
	4
	t drinks | good | neutral ; service | bad | negative

如果我们可以准确地知道哪一个beam最好

|           | 14res  | 14lap  | 15res  | 16res  |
| --------- | ------ | ------ | ------ | ------ |
| precision | 0.8373 | 0.7846 | 0.7510 | 0.8266 |
| recall    | 0.8380 | 0.7495 | 0.7711 | 0.8658 |
| f1-score  | 0.8377 | 0.7667 | 0.7609 | 0.8457 |

最好的beam是哪一个

|      | 14res  | 14lap  | 15res  | 16res  |
| ---- | ------ | ------ | ------ | ------ |
| 0    | 61.88% | 54.57% | 54.04% | 61.96% |
| 1    | 12.97% | 13.11% | 8.07%  | 10.74% |
| 2    | 7.58%  | 6.71%  | 9.01%  | 8.90%  |
| 3    | 9.78%  | 12.20% | 12.73% | 9.20%  |
| 4    | 7.78%  | 13.41% | 16.15% | 9.20%  |

上述结果运行以下命令得到

	bash/do_extraction.sh -b 14res_test.json -d data/t5/15res/test.json -c 3 -m {YOUR_OUTPUT_DIR}/model/dataset=15res,b=test,seed=42 -o {YOUR_OUTPUT_DIR}


## 标注相对好坏

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/reward-model.png" alt="img" style="zoom:40%;" />

1. 在已有的训练集上训练模型，然后对无标记数据打标签
2. 搭建标注系统，如https://github.com/davidjurgens/potato ，标注时添加原句子的中文翻译
3. 阅读现有案例
4. 初步标注1000条数据（14res）
5. 合并人类排序
6. 训练奖励模型

对无标记数据打标签

	bash/do_extraction.sh -b yelp2023.json -d {yelp2023_dir} -c 3 -m {YOUR_OUTPUT_DIR}/model/dataset=15res,b=test,seed=42 -o {YOUR_OUTPUT_DIR} -v v2

## 利用人类反馈来增强模型性能

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rlhf/rlhf.png" alt="img" style="zoom:40%;" />

1. 使用PPO的框架训练抽取器，参考 https://github.com/juncongmoo/chatllama