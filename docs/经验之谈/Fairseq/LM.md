# 神经语言建模

## 语言模型采样
fairseq已经预训练了部分语言模型，可以采用PyTorch Hub直接调用，详见：https://github.com/facebookresearch/fairseq/blob/main/examples/language_model/README.md

语言模型的几个重要功能：
- 采样（sample）：在原始的长序列上任意捕获的子序列
- 计算句子困惑度（Compute perplexity for a sequence）

## 使用 fairseq 训练一个Transformer语言模型
1. 预处理数据
- 下载并准备数据
    - 如： WikiText-103 数据集
- 预处理/二值化
```bash
TEXT = wikitext-103
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir data-bin/wikitext-103 \
    --workers 20
```

2. 训练语言模型
训​​练一个基本的 LM（假设有 2 个 GPU）：
```bash
fairseq-train --task language_modeling \
    data-bin/wikitext-103 \
    --save-dir checkpoints/transformer_wikitext-103 \
    --arch transformer_lm --share-decoder-input-output-embed \
    --dropout 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --tokens-per-sample 512 --sample-break-mode none \
    --max-tokens 2048 --update-freq 16 \
    --fp16 \
    --max-update 50000
```
如果内存不足，请尝试减少 ```--max-tokens```（每批次的最大标记数）或 ```--tokens-per-sample```（最大序列长度）。您还可以调整  ```--update-freq```以累积梯度并在不同数量的 GPU 上模拟训练。

3. 评估语言模型
```bash
fairseq-eval-lm data-bin/wikitext-103 \
    --path checkpoints/transformer_wiki103/checkpoint_best.pt \
    --batch-size 2 \
    --tokens-per-sample 512 \
    --context-window 400
# | Evaluated 245569 tokens in 56.1s (4379.02 tokens/s)
# | Loss: 3.4164, Perplexity: 30.46
```
注意：该--context-window选项控制在计算困惑度时为每个标记提供多少上下文。当窗口大小为 0 时，数据集被分成长度为 512 的段，并且通常在每个段上计算困惑度。然而，这会导致更糟糕（更高）的困惑，因为在每个段中出现较早的标记具有较少的条件。当使用最大窗口大小时（在本例中为 511），然后我们计算完全以 511 个上下文标记为条件的每个标记的困惑度。这会显着减慢评估速度，因为我们必须为数据集中的每个标记运行单独的前向传递，但会产生更好（更低）的困惑。