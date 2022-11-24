## Sentencepiece
### 简介
sentencepiece由谷歌将一些词-语言模型相关的论文进行复现，开发了一个开源工具——训练自己领域的sentencepiece模型，该模型可以代替预训练模型(BERT,XLNET)中词表的作用。开源代码地址为：https://github.com/google/sentencepiece。

### 原理
提供四种关于词的切分方法。这里跟中文的分词作用是一样的，但从思路上还是有区分的。通过使用我感觉：在中文上，就是把经常在一起出现的字组合成一个词语；在英文上，它会把英语单词切分更小的语义单元，减少词表的数量。

例如“机器学习领域“这个文本，按jieba会分“机器/学习/领域”，但你想要粒度更大的切分效果，如“机器学习/领域”或者不切分，这样更有利于模型捕捉更多N-gram特征。为实现这个，你可能想到把对应的大粒度词加到词表中就可以解决，但是添加这类词是很消耗人力。然而对于该问题，sentencepiece可以得到一定程度解决，甚至完美解决你的需求。

模型在训练中主要使用统计指标，比如出现的频率，左右连接度等，还有困惑度来训练最终的结果。了解算法细节可以去githup上查看相关论文。

## 功能
1. 训练模型：用C语言实现的，可编成二进程程序执行，训练结果是生成一个model和一个词典文件。
2. 使用模型：同时支持二进制程序和Python调用两种方式，训练完生成的词典数据是明文，可编辑，因此也可以用任何语言读取和使用。

## 安装
```
pip install sentencepiece
```



Reference
https://github.com/google/sentencepiece
https://zhuanlan.zhihu.com/p/159200073
https://github.com/google/sentencepiece/blob/master/python/README.md