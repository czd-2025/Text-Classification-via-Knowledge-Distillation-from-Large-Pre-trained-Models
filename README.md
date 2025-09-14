# Knowledge Distillation for Text Classification based on Large Models 
# 基于大模型知识蒸馏的政策文本分类

本项目旨在利用大型预训练模型进行知识蒸馏，实现高效且精准的政策文本分类。我们提供了一套开箱即用的中文文本分类解决方案，基于 PyTorch 实现，并支持多种 BERT 变体及 ERNIE 等模型。

## 技术概览

*   **任务：** 政策文本分类
*   **语言：** 中文
*   **基础框架：** PyTorch
*   **核心思想：** 知识蒸馏

## 模型架构

*   **教师模型 (Teacher Models):**  我们计划支持多种强大的教师模型，例如 Gemini 2.5, GPT, DeepSeek 等（具体模型选择取决于实验效果）。
*   **学生模型 (Student Models):**  本项目主要采用 BERT 及其变体作为学生模型，旨在通过知识蒸馏提升其在特定任务上的性能。

## 数据流程

模型的详细介绍和数据流动过程涉及论文隐私，敬请谅解。

## 硬件与训练

*   **GPU:** NVIDIA GeForce RTX 4060
*   **训练时间:**
    *   基础 BERT 模型：约 30 分钟
    *   加入 CNN 层的 BERT 模型：可能长达 2-3 天 (取决于具体配置)

## 环境配置

*   **操作系统:** Windows 11
*   **Python:** 3.10.18
*   **CUDA:** 12.6
*   **PyTorch:** 2.5.1+cu121
*   **依赖库:**
    *   `tqdm`
    *   `sklearn`
    *   `tensorboardX`
    *   `pytorch_pretrained_bert` (本项目已上传预训练代码，因此不再需要单独安装此库)

## 数据集

*   **类型：** 中文政策文本数据集
*   **预处理：** 基于 BERT 谷歌发布的相关代码修改，以字为单位将数据输入模型。
*   **数据集来源：** 默认使用 THUCNews 数据集，您可以轻松替换为自定义的政策语句数据集。
*   **类别示例：**
    *   人才培养
    *   资金投入
    *   科技投入
    *   公共服务
    *   设施建设
    *   目标规划
    *   法规管制
    *   金融支持
    *   政策支持
    *   产权保护

*   **数据集划分：** 请根据您自身数据集的格式进行调整。

## 实验结果

下表展示了不同模型的分类精度 (Accuracy)：

| 模型        | Accuracy | 备注                                                       |
| ----------- | -------- | ---------------------------------------------------------- |
| BERT        |  （待补充）  | 原始 BERT 模型                                                |
| ERNIE       |  （待补充）  | 实验结果表明，ERNIE 在本项目中表现不如 BERT (原因待分析)                 |
| BERT+CNN    |  （待补充）  | 将 BERT 作为 Embedding 层，结合 CNN 模型                              |
| BERT+RNN    |  （待补充）  | 将 BERT 作为 Embedding 层，结合 RNN 模型                              |
| BERT+RCNN   |  （待补充）  | 将 BERT 作为 Embedding 层，结合 RCNN 模型                             |
| BERT+自定义  |  （待补充）  | 将 BERT 作为 Embedding 层，结合 自定义 模型                            |

**结论：** 实验初步表明，原始 BERT 模型已经取得了较好的效果。将 BERT 作为 Embedding 层并与其他模型结合，性能反而有所下降。后续将尝试长文本上的效果对比。

## 预训练语言模型

本项目需要预训练语言模型，请将模型文件放置在以下目录下：

*   **BERT 模型:** `bert_pretrain/` (包含 `pytorch_model.bin`, `bert_config.json`, `vocab.txt` 三个文件)
*   **ERNIE 模型:** `ERNIE_pretrain/` (包含 `pytorch_model.bin`, `bert_config.json`, `vocab.txt` 三个文件)

预训练模型下载地址：

bert_Chinese: 模型 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz

词表 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt

来自这里

备用：模型的网盘地址：https://pan.baidu.com/s/1qSAD5gwClq7xlgzl_4W3Pw

ERNIE_Chinese: http://image.nghuyong.top/ERNIE.zip

来自这里

备用：网盘地址：https://pan.baidu.com/s/1lEPdDN1-YQJmKEd_g9rLgw

解压后，按照上面说的放在对应目录下，文件名称确认无误即可。


使用说明

下载好预训练模型就可以跑了。

# 训练并测试：
# bert
python run.py --model bert

# bert + 其它
python run.py --model bert_CNN

# ERNIE
python run.py --model ERNIE

参数

模型都在models目录下，超参定义和模型定义在同一文件中。


未完待续

    封装预测政策工具所属类别功能
