from datasets import load_dataset
import json
import os

DATASET_CATEGORY = "forget01"
OUTPUT_DIR = "../../assets/datasets/tofu"


def save_dataset_to_json(dataset_split, output_dir, split_name):
    """将指定数据集划分保存为 JSON 文件"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split_name}.json")

    # 转为普通 Python list 格式（以便保存）
    data_list = [dict(item) for item in dataset_split]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=2)

    print(f"✅ 已保存: {output_path} （共 {len(data_list)} 条数据）")


def main():
    print("🚀 开始下载 TOFU 数据集（locuslab/TOFU）...")
    # 下载完整版本（也可以改成 "forget05"、"forget10" 等）
    dataset = load_dataset("locuslab/TOFU", f"{DATASET_CATEGORY}")

    print("📦 数据集结构:")
    print(dataset)

    for split_name in dataset.keys():
        save_dataset_to_json(dataset[split_name], OUTPUT_DIR, DATASET_CATEGORY)

    print("🎉 所有数据已下载并保存完成！")


if __name__ == "__main__":
    main()
