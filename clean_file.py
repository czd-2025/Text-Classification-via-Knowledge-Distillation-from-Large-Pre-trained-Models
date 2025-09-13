import re
import json
from tqdm import tqdm


def clean_file(input_path, output_path):
    """
    读取一个包含脏概率数据的文件，清理后写入一个新文件。
    同时，记录并返回处理过程中发现的所有错误。
    """
    print(f"开始清理文件: {input_path}")
    cleaned_lines = []
    # --- 修改点 1: 新增一个列表来记录错误信息 ---
    errors_found = []

    with open(input_path, 'r', encoding='utf-8') as f:
        # --- 修改点 2: 使用 enumerate 获取行号 (从1开始) ---
        for line_num, line in enumerate(tqdm(f, desc=f"正在处理 {input_path}"), 1):
            try:
                # 尝试按制表符分割，通常是 "文本\t标签\t概率列表"
                parts = line.strip().split('\t')
                # 检查格式是否正确，这是最常见的错误源
                if len(parts) < 3:
                    # --- 修改点 3: 记录格式错误信息，而不是静默跳过 ---
                    error_info = {
                        "line_num": line_num,
                        "content": line.strip(),
                        "error": "格式错误：该行无法被正确分割为3部分"
                    }
                    errors_found.append(error_info)
                    continue

                text, label, prob_str = parts[0], parts[1], parts[2]

                # 核心清洗逻辑：
                # 1. 使用正则表达式找出所有合法的数字（包括整数和浮点数）
                numbers_found = re.findall(r'-?\d+\.?\d*', prob_str)

                # 2. 将找到的数字字符串转换为浮点数
                cleaned_probs = [float(num) for num in numbers_found]

                # 3. 将清理后的浮点数列表转换回标准的JSON字符串格式
                cleaned_prob_str = json.dumps(cleaned_probs)

                # 4. 重新组合成一行
                new_line = f"{text}\t{label}\t{cleaned_prob_str}"
                cleaned_lines.append(new_line)

            except Exception as e:
                # --- 修改点 4: 捕获其他异常，并记录详细信息 ---
                error_info = {
                    "line_num": line_num,
                    "content": line.strip(),
                    "error": f"发生意外错误: {e}"
                }
                errors_found.append(error_info)
                continue

    # 将所有清理好的行写入新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            f.write(line + '\n')

    print(f"清理完成！已将结果保存到: {output_path}")

    # --- 修改点 5: 返回找到的错误列表 ---
    return errors_found


if __name__ == '__main__':
    # --- 您需要配置的路径 ---
    files_to_clean = {
        'THUCNews/data/train.txt': 'THUCNews/data/train_cleaned.txt',
        'THUCNews/data/dev.txt': 'THUCNews/data/dev_cleaned.txt',
        'THUCNews/data/test.txt': 'THUCNews/data/test_cleaned.txt'
    }
    # -------------------------

    # --- 修改点 6: 创建一个列表来汇总所有文件的错误 ---
    total_errors = {}

    for original_file, cleaned_file in files_to_clean.items():
        # clean_file 函数现在会返回错误列表
        errors = clean_file(original_file, cleaned_file)
        if errors:
            total_errors[original_file] = errors
        print("-" * 40)  # 打印分割线，让输出更清晰

    # --- 修改点 7: 在所有任务结束后，打印最终的错误汇总报告 ---
    print("\n==================== 错误汇总报告 ====================")
    if not total_errors:
        print("🎉 恭喜！所有文件均已成功处理，未发现任何格式错误。")
    else:
        print(f"处理完成，但在以下文件中发现了错误：")
        for filename, errors in total_errors.items():
            print(f"\n📄 文件: {filename} (共 {len(errors)} 个错误)")
            for error in errors:
                print(f"  - 错误行号: {error['line_num']}")
                print(f"    错误内容: '{error['content']}'")
                print(f"    错误原因: {error['error']}")
    print("======================================================")

    print("\n所有文件均已清理完毕！")
    print("下一步：请修改您主项目中的 Config 类，将文件名指向新的 '_cleaned.txt' 文件。")