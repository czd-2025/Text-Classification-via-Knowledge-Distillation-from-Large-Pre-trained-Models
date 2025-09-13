# predict_and_visualize.py

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from models.bert_gru_attention import Model, Config  # 确保从正确的路径导入你的模型

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
def visualize_attention(model, config, text):
    """
    可视化给定文本的注意力权重。

    Args:
        model:  训练好的 bert-gru-attention 模型.
        config: 配置对象.
        text:  输入的文本 (string).
    """
    model.eval()  # 设置为评估模式
    tokenizer = config.tokenizer
    device = config.device
    tokens = tokenizer.tokenize(text)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    seq_len = len(token_ids)
    mask = [1] * seq_len
    pad_size = config.pad_size  # 从config中获取pad_size

    if seq_len < pad_size:
        mask += [0] * (pad_size - seq_len)
        token_ids += ([0] * (pad_size - seq_len))
    else:
        token_ids = token_ids[:pad_size]
        mask = mask[:pad_size]
        seq_len = pad_size

    # 转换为 tensors
    token_ids_tensor = torch.LongTensor([token_ids]).to(device)
    mask_tensor = torch.LongTensor([mask]).to(device)
    # 模型输入是元组 (ids, seq_len, mask)
    inputs = (token_ids_tensor, None, mask_tensor)

    # 获取 attention weights
    with torch.no_grad():
        # 修改点：确保 forward 函数返回 attention weights
        outputs, attention_weights = model(inputs, return_attention=True)

    attention_weights = attention_weights.squeeze().cpu().numpy()

    # 去除 [CLS] 和 [SEP] 符号的权重
    tokens = tokens[1:-1]
    attention_weights = attention_weights[1:len(tokens)+1]

    # 3. 可视化
    plt.figure(figsize=(12, 6))  # 调整图像大小
    sns.heatmap(attention_weights.reshape(1, -1),
                xticklabels=tokens,
                yticklabels=['Attention'],
                cmap="YlGnBu",  # 使用更清晰的颜色映射
                linewidths=.5,
                annot=True,  # 显示数值
                fmt=".2f")  # 格式化数值
    plt.title('Attention Weights Visualization')  # 添加标题
    plt.xlabel('Tokens')  # 添加 x 轴标签
    plt.ylabel('Layer')  # 添加 y 轴标签
    plt.xticks(rotation=45, ha="right")  # 旋转 x 轴标签
    plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
    plt.show()


def predict_single_sentence(model, config, text):
    """
    对单个句子进行预测并可视化其注意力权重。
    """
    # 将模型设置为评估模式
    model.eval()

    # 1. 文本预处理
    tokenizer = config.tokenizer
    device = config.device
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]

    # 截断或填充到pad_size
    if len(tokens) < config.pad_size:
        seq_len = len(tokens)
        mask = [1] * seq_len + [0] * (config.pad_size - seq_len)
        token_ids = tokenizer.convert_tokens_to_ids(tokens) + [0] * (config.pad_size - seq_len)
    else:
        seq_len = config.pad_size
        mask = [1] * config.pad_size
        tokens = tokens[:config.pad_size]
        token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # 转换为Tensor
    token_ids_tensor = torch.LongTensor([token_ids]).to(device)
    mask_tensor = torch.LongTensor([mask]).to(device)
    # 模型输入是元组 (ids, seq_len, mask)
    # 注意：这里的seq_len参数在你的模型中没有被使用，但为了保持项目统一性，我们传入
    inputs = (token_ids_tensor, None, mask_tensor)

    # 2. 模型预测，并请求返回Attention
    with torch.no_grad():
        # 调用forward时，设置 return_attention=True
        outputs, attention_weights = model(inputs, return_attention=True)

    # 3. 解析预测结果
    pred_class_idx = torch.argmax(outputs).item()
    pred_class = config.class_list[pred_class_idx]

    print("=" * 30)
    print(f"输入句子: {text}")
    print(f"预测类别: {pred_class}")
    print("=" * 30)

    # 4. 可视化Attention
    visualize_attention(model, config, text)


if __name__ == '__main__':
    # --- 配置 ---
    dataset = 'THUCNews'  # 你的数据集名称
    config = Config(dataset)
    model = Model(config).to(config.device)

    # --- 加载已训练好的模型 ---
    # 确保你的模型已经训练并保存在了正确的路径
    try:
        model.load_state_dict(torch.load(config.save_path, map_location=config.device))
        print("模型加载成功！")
    except FileNotFoundError:
        print(f"错误：找不到模型文件 {config.save_path}")
        print("请先运行 train_eval.py 训练模型。")
        exit()

    # --- 输入你想测试的句子 ---
    sentence1 = "建立以创新价值、能力、贡献为导向的科技人才评价体系。"
    sentence2 = "对运营单位的数据安全保障能力、数据产品质量、市场效益等进行绩效评价。"

    predict_single_sentence(model, config, sentence1)
    predict_single_sentence(model, config, sentence2)