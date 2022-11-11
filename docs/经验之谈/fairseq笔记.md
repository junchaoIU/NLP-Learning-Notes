## Faieseq 
Fairseq是一个用PyTorch编写的序列建模工具包，它允许研究人员和开发人员训练用于翻译、摘要、语言建模和其他文本生成任务的定制模型。本系列笔记主要以翻译官方文档为主，附带一些个人的学习记录。官方教程连接：https://fairseq.readthedocs.io/en/latest/getting_started.html。

环境配置
```
git clone https://github.com/pytorch/fairseq.git 
cd fairseq
pip install --editable ./

# 安装失败的可能原因：版本问题
PyTorch version >= 1.10.0
Python version >= 3.8
```


tokenization and BPE
```
# 下载Moses, tokenization采用Moses
echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

# 脚本地址
SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000

# 数据集下载路径
URL="http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz"
# 数据集名称
GZ=de-en.tgz

# 监测Moses scripts是否正确
if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

# 源语言类型
src=de
# 目标语言类型
tgt=en
# 翻译类型
lang=de-en
prep=iwslt14.tokenized.de-en
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

# 将数据集下载到orig路径下
echo "Downloading data from ${URL}..."
cd $orig
wget "$URL"

# 判断数据集是否成功下载
if [ -f $GZ ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

# 解压数据集
tar zxvf $GZ
cd ..

# 训练集tokenization
echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat $orig/$lang/$f | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    sed -e 's/<title>//g' | \LC 
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done

# 训练集清洗
perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 175
for l in $src $tgt; do
    perl $LC < $tmp/train.tags.$lang.clean.$l > $tmp/train.tags.$lang.$l
done

# 验证集和测试集清洗
echo "pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/$lang/IWSLT14.TED*.$l.xml`; do
    fname=${o##*/}
    f=$tmp/${fname%.*}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $f
    echo ""
    done
done


echo "creating train, valid, test..."
# 分割训练集和验证集
for l in $src $tgt; do
    awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.de-en.$l > $tmp/valid.$l
    awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.de-en.$l > $tmp/train.$l

    # 合成测试集
    cat $tmp/IWSLT14.TED.dev2010.de-en.$l \
        $tmp/IWSLT14.TEDX.dev2012.de-en.$l \
        $tmp/IWSLT14.TED.tst2010.de-en.$l \
        $tmp/IWSLT14.TED.tst2011.de-en.$l \
        $tmp/IWSLT14.TED.tst2012.de-en.$l \
        > $tmp/test.$l
done

# 训练集目录
TRAIN=$tmp/train.en-de

BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

# 三个数据集都进行BPE分词
for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done
```






命令解析：
```bash
--task，意思是任务类型，默认是“translation”（翻译）；包括translation_from_pretrained_bart等Fairseq自带的任务类型；
--arch，意思是model architecture，模型结构，可选项包括，transformer，lstm等；这里采用自带的结构fastcorrect 
 --lr 学习率，为初始学习率，后续可能被--lr-scheduler修改；--lr-scheduler 是lr更新计划，这里采用inverse_sqrt 方法；
--dropout 字面意思就是丢弃，这里指的是在训练模型时，丢弃一部分数据来防止过拟合；
--optimizer adam --adam-betas '(0.9, 0.98)' 参数优化策略；
--criterion fc_loss 训练准则；
--max-tokens 9000 一个batch最大的token数量；
--save-dir $SAVE_DIR 存储checkpoints的路径，checkpoint即模型；
--user-dir $EXP_HOME/FastCorrect 一个包含扩展的python模块，这里的扩展是指模型结构或者任务，和task是相对的，一般不适用官方规定的arch时需要手动设置这个路径；
--max-epoch 30 当达到这个30个epoch的时候，停止训练；
--update-freq 4 参数更新频率，每4个batch更新参数；
--fp16 使用FP16
--num-workers 8 8个子线程用于load数据；
```
举例：

```
PRETRAIN=mbart.cc25 # fix if you moved the downloaded checkpoint
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

fairseq-train path_2_data \
  --encoder-normalize-before --decoder-normalize-before \
  --arch mbart_large --layernorm-embedding \
  --task translation_from_pretrained_bart \
  --source-lang en_XX --target-lang ro_RO \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler polynomial_decay --lr 3e-05 --warmup-updates 2500 --total-num-update 40000 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 1024 --update-freq 2 \
  --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 10 --no-epoch-checkpoints \
  --seed 222 --log-format simple --log-interval 2 \
  --restore-file $PRETRAIN \
  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
  --langs $langs \
  --ddp-backend legacy_ddp
```

预训练
```
fairseq-preprocess \
  --source-lang "source" \
  --target-lang "target" \
  --trainpref "${TASK}/train.bpe" \
  --validpref "${TASK}/val.bpe" \
  --destdir "${TASK}-bin/" \
  --workers 60 \
  --srcdict dict.txt \
  --tgtdict dict.txt;
```