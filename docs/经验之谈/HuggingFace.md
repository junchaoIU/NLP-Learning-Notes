## Transformer基础功能 pipeline
 Transformers 库中最基本的对象是 pipeline() 函数。它将模型与其必要的预处理和后处理步骤连接起来，使我们能够通过直接输入任何文本并获得最终的答案。目前支持的pipeline任务有：
 - feature-extraction (get the-ector representation of a text)
- fill-mask
- ner (named entity recognition)
- question-answering
- sentiment-analysis
- summarization
- text-generation
- translation
- zero-shot-classification

```python
 from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("I've been waiting for a HuggingFace course my whole life."))

# 输出
# [{'label': 'POSITIVE', 'score': 0.9598047137260437}]
# [{'label': 'POSITIVE', 'score': 0.9598047137260437}, {'label': 'NEGATIVE','score': 0.9994558095932007}]
```

## pipeline特定模型
前面的示例使用了默认模型，但您也可以从 Hub 中选择特定模型以在特定任务的pipeline中使用 
以 distilgpt2 模型为例，以下是如何在与以前相同的pipeline中加载它：

```python
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)

# [{'generated_text': 'In this course, we will teach you how to manipulate the world and move your mental and physical capabilities to your advantage.'},
# {'generated_text': 'In this course, we will teach you how to become an expert and practice realtime, and with a hands on experience on both real time and real'}]
```

## fill-mask任务
此任务的想法是填充给定文本中的空白。
```python
from transformers import pipeline
unmasker = pipeline("fill-mask")
print(unmasker("This course will teach you all about <mask> models.", top_k=2))

# [{'sequence': 'This course will teach you all about mathematical models.',
# 'score': 0.19619831442832947,
# 'token': 30412,
# 'token_str': ' mathematical'},
# {'sequence': 'This course will teach you all about computational models.',
# 'score': 0.04052725434303284,
# 'token': 38163,
# 'token_str': ' computational'}]
```
top_k 参数控制要显示的结果有多少种。请注意，这里模型填充了特殊的< mask >词，它通常被称为掩码标记。其他掩码填充模型可能有不同的掩码标记，因此在探索其他模型时要验证正确的掩码字是什么。检查它的一种方法是查看小组件中使用的掩码。