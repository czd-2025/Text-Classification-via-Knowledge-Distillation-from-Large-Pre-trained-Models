# Knowledge Distillation for Text Classification based on Large Models 
基于大模型知识蒸馏的文本分类——政策文本分类


中文文本分类，Bert及其变体，ERNIE，基于pytorch，开箱即用。

教师模型：Gemini 2.5；GPT；deepseek...

学生模型：Bert及其变体


介绍

模型介绍、数据流动过程：论文隐私，请谅解

显卡：NVIDIA GeForce RTX 4060 ， 训练时间：30分钟左右，加入CNN层的话训练时间可能长达2-3天。


环境

Windows 11


python	3.10.18


cuda	12.6


pytorch	2.5.1+cu121


tqdm


sklearn


tensorboardX


pytorch_pretrained_bert(预训练代码也上传了, 不需要这个库了)


中文数据集

根据Bert谷歌发布的相关代码修改

THUCNews可改政策语句

数据以字为单位输入模型。


类别：人才培养；资金投入；科技投入；公共服务；设施建设；目标规划；法规管制；金融支持；政策支持；产权保护等。

数据集划分：
更换自己的数据集

    按照我数据集的格式来格式化你的中文数据集。

效果

模型 	acc 	备注

bert 	

ERNIE 		说好的中文碾压bert呢（怎么变差啦）

bert_CNN 	 	bert + CNN

bert_RNN 		bert + RNN

bert_RCNN 		bert + RCNN

bert_DPCNN 	 	bert + DPCNN

原始的bert效果就很好了，把bert当作embedding层送入其它模型，效果反而降了，之后会尝试长文本的效果对比。

预训练语言模型

bert模型放在 bert_pretain目录下，ERNIE模型放在ERNIE_pretrain目录下，每个目录下都是三个文件：

    pytorch_model.bin
    
    bert_config.json
    
    vocab.txt

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
