# 中文自然语言处理资源笔记
> 个人收集的自用及备用的自然语言处理相关资源

## 开源Python库

| 项目                                  | 地址                                            | 简介                                                         |
|:-------------------------------------| ----------------------------------------------- | ------------------------------------------------------------ |
| jieba分词                             | https://github.com/fxsjy/jieba                  | 中文分词库                                                   |
| 中文信息抽取工具                      | https://github.com/fighting41love/cocoNLP       | 从中文文本数据中抽取出结构化的信息，如时间、手机号、运营商、邮箱、地址、人名、身份证 |
| LTP（Language Technology Platform）   | https://github.com/HIT-SCIR/ltp                 | 提供了一系列中文自然语言处理工具，用户可以使用这些工具对于中文文本进行分词、词性标注、句法分析等等工作 |
| 中文地址提取工具                      | https://github.com/shibing624/addressparser     | 支持中国三级区划地址（省、市、区）提取和级联映射，支持地址目的地热力图绘制。适配python2和python3 |
| 中文公司名称分词工具                  | https://github.com/shibing624/companynameparser | 支持公司名称中的地名，品牌名（主词），行业词，公司名后缀提取 |
| 汉字数字(中文数字)-阿拉伯数字转换工具 | https://github.com/Wall-ee/chinese2digits       | 是一个将中文数字（大写数字） 转化为阿拉伯数字的工具          |
| HarvestText                          | https://github.com/blmoistawinde/HarvestText    | 是一个专注无（弱）监督方法，能够整合领域知识（如类型，别名）对特定领域文本进行简单高效地处理和分析的库。适用于许多文本预处理和初步探索性分析任务，在小说分析，网络文本，专业文献等领域都有潜在应用价值 |
|                                      |                                                 |                                                              |
|                                      |                                                 |                                                              |

## 知识图谱相关

| 项目                    | 地址                                                        | 简介                                                         |
| ----------------------- | ----------------------------------------------------------- | ------------------------------------------------------------ |
| 文档图谱信息可视化      | https://github.com/liuhuanyong/TextGrapher                  | 输入一篇文档，将文档进行关键信息提取，进行结构化，并最终组织成图谱组织形式，形成对文章语义信息的图谱化展示。 |
| 京东GoodsKG             | https://github.com/liuhuanyong/ProductKnowledgeGraph        | 基于京东网站的商品上下级概念，商品品牌之间关系，商品描述维度等知识库，基于该知识库可以支持商品属性库构建，商品销售问答，品牌物品生产等知识查询服务，也可用于情感分析等下游应用． |
| 思知知识图谱            | https://github.com/ownthink/KnowledgeGraphData              | 史上最大规模1.4亿中文知识图谱开源下载，知识图谱，通用知识图谱，融合了两千五百多万的实体，拥有亿级别的实体属性关系。 |
| stock-knowledge-graph   | https://github.com/lemonhu/stock-knowledge-graph            | （neo4j）利用网络上公开的数据构建一个小型的证券知识图谱/知识库 |
| 事件三元组抽取          | https://github.com/liuhuanyong/EventTriplesExtraction       | 基于依存句法与语义角色标注的事件三元组抽取，可用于文本理解如文档主题链，事件线等应用。内置LTP、百度DDParser和规则模版的三种抽取方式 |
| 中文人物知识图谱构建    | https://github.com/liuhuanyong/PersonRelationKnowledgeGraph | 中文人物关系知识图谱项目,内容包括中文人物关系图谱构建,基于知识库的数据回标,基于远程监督与bootstrapping方法的人物关系抽取,基于知识图谱的知识问答等应用. |
| awesome-knowledge-graph | https://github.com/husthuke/awesome-knowledge-graph         | 整理知识图谱相关学习资料，提供系统化的知识图谱学习路径。     |



## 语料&数据集

| 项目                                 | 地址                                                 | 简介                                                         |
| ------------------------------------ | ---------------------------------------------------- | ------------------------------------------------------------ |
| ChineseNlpCorpus                     | https://github.com/SophonPlus/ChineseNlpCorpus       | 搜集、整理、发布 中文 自然语言处理 语料/数据集, 包含情感/观点/评论 倾向性分析、中文命名实体识别、推荐系统、FAQ 问答系统多个领域的数据集|
| 公司名语料库（Company-Names-Corpus） | https://github.com/wainshine/Company-Names-Corpus    | 公司名语料库。机构名语料库。公司简称,缩写,品牌词,企业名。可用于中文分词、机构名实体识别。 |
| 微信公众号语料库                     | https://github.com/nonamestreet/weixin_public_corpus | 部分网络抓取的微信公众号的文章，已经去除HTML，只包含了纯文本。 |
| 百度知道问答语料库                   | https://github.com/liuhuanyong/MiningZhiDaoQACorpus  | 百度知道问答语料库，包括超过580万的问题，938万的答案，5800个分类标签。基于该问答语料库，可支持多种应用，如闲聊问答，逻辑挖掘。 |
| 多语言音频数据                       | https://voice.mozilla.org/en/datasets                | 多种语言音频数据，包括来自42,000名贡献者超过1,400小时的语音样本，涵github |
| 中文突发事件语料库                   | https://github.com/shijiebei2009/CEC-Corpus          | 中文突发事件语料库是由上海大学（语义智能实验室）所构建。根据国务院颁布的《国家突发公共事件总体应急预案》的分类体系，从互联网上收集了5类（地震、火灾、交通事故、恐怖袭击和食物中毒）突发事件的新闻报道作为生语料，然后再对生语料进行文本预处理、文本分析、事件标注以及一致性检查等处理，最后将标注结果保存到语料库中，CEC合计332篇。 |
| dh_msra                              |[下载地址](https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/dh_msra/intro.ipynb)|  5 万多条中文命名实体识别标注数据（包括地点、机构、人物）  |

## 词表

| 项目      | 地址                                  | 简介                             |
| --------- | ------------------------------------- | -------------------------------- |
| multistop | https://github.com/hidadeng/multistop | 停用词表，支持中英法德等15种语言 |
|           |                                       |                                  |



## 其他可能有帮助的研究

| 项目             | 地址                                                  | 简介                                                         |
| ---------------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| 事理知识抽取研究 | https://github.com/liuhuanyong/ComplexEventExtraction | 中文复合事件抽取，包括条件事件、因果事件、顺承事件、反转事件等事件抽取，并形成事理图谱。 |
| 领域情感词典构建 | https://github.com/hidadeng/wordexpansion             | 使用SO_PMI互信息算法简单快速构建不同领域(手机、汽车等)的专业情感词典 |