# NLPCC2016-Task4

## GuideLine

[http://tcci.ccf.org.cn/conference/2016/](http://tcci.ccf.org.cn/conference/2016/)

#### Introduction

NLPCC-ICCPOL 2016 Shared Task: Stance Detection in Chinese Microblogs

该任务旨在评估微博文本的立场检测技术。 立场检测任务旨在自动确定微博文本的作者是否支持给定目标、反对给定目标或两者都不支持。 请注意，给定的目标可能不会出现在微博文本中。 这意味着立场检测不同于传统的目标/方面情感分析。



考虑目标-微博对：

Target: 俄罗斯在叙利亚的空袭行动 

Weibo: 9月30日开始至今，俄在叙利亚发起空隙，共死亡1331人，其中403人是一般的民众, 其中的三分之一是无辜平民陪葬。 

据观察，微博文本的作者反对给定的目标。 立场检测的目的是评估检测每个微博文本作者立场的技术。 通常，系统需要识别焦点文本中可能不存在的相关信息。 例如，如果一个人强调平民的死亡，那么他或她很可能反对俄罗斯的空袭。 因此，我们提供了与每个目标相关的域语料库，系统可以从中收集有用的信息以进行立场检测。 本次评估有两个任务。 **任务 A 是一项强制性任务，每个参与者都必须提交此任务的结果。 但是，任务 B 是可选任务，参与者可以自行决定是否提交该任务的结果。**



### Tasks

Task A (监督/半监督框架): This task aims to detect stance towards five targets such as "俄罗斯在叙利亚的空袭行动" and "《太阳的后裔》热播". **A total of 3,000 labeled instances for all targets** will be provided as training data as well as a large amount of unlabeled data. 



Task B (无监督框架): This task aims to detect stance towards another two targets. No training data will be provided. Instead, a large set of Weibo texts associated with each target, without any stance annotation, will be provided. 



### Labels

The possible stance labels are described as follows: 

- FAVOR: The author is in favor of the target (e.g., directly or indirectly by supporting someone/something, by opposing or criticizing someone/something opposed to the target, or by echoing the stance of somebody else).  
- AGAINST: The author is against the target (e.g., directly or indirectly by opposing or criticizing someone/something, by supporting someone/something opposed to the target, or by echoing the stance of somebody else).  
- NONE: none of the above. 



### Data Format

提交格式：测试数据文件与训练文件具有相同的格式，除了类标签对于所有实例都是“UNKNOWN”。 参与系统需要将“UNKNOWN”替换为其预测的类以创建提交文件。



The format of training data file is: 

<ID> <tab> <Target> <tab> <Weibo> <tab> <Stance> 

<ID> is an internal identification number; 

<Target> is the given target; 

<Tweet> is a Weibo text; 

<Stance> is the stance label. 



### 评价方法

评价：以F-score（FAVOR）和F-score（AGAINST）的宏观平均数作为底线评价指标。 每个参与者每项任务只能提交一个运行结果。

