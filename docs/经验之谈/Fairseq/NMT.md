## 神经机器翻译
fairseq 提供了使用预训练模型和训练新模型的功能

fairseq已经预训练了部分机器翻译模型（主要是Transformer），可以采用PyTorch Hub直接调用，详见：https://github.com/facebookresearch/fairseq/blob/main/examples/translation/README.md#example-usage-torchhub

## 训练新NMT模型
1. 下载并预处理数据
- 准备数据
    - 如：iwslt14.tokenized.de-en
- 预处理/二值化数据
```bash
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess 
    --source-lang de 
    --target-lang en \
    --trainpref $TEXT/train \
    --validpref $TEXT/valid \
    --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```
2. 训练一个 Transformer 翻译模型
```bash
CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \ --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric
```

3. 评估训练好的模型
```bash
fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/checkpoint_best.pt \
    --batch-size 128 \
    --beam 5 \
    --remove-bpe | tee /tmp/gen.out

# | Translated 3003 sentences (96311 tokens) in 166.0s (580.04 tokens/s)
# | Generate test with beam=5: BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)

# Compute BLEU score
grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
fairseq-score --sys /tmp/gen.out.sys --ref /tmp/gen.out.ref
# BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)
```