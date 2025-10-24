import os
import json
import subprocess
import time
import pandas as pd
from beijing_data_download_config import USER_KEY, DATASETS

OUTPUT_DIR = "../../assets/datasets/beijing_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_page_with_curl(url):
    """使用 curl 获取单页数据，自动重试"""
    for attempt in range(3):
        try:
            result = subprocess.run(
                ["curl", "-s", "--connect-timeout", "15", url],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True
            )
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"  ⚠️ 重试 {attempt + 1}/3: {e}")
            time.sleep(2)
    raise RuntimeError(f"无法获取 URL: {url}")


def download_dataset(name, content_id, page_size=100):
    print(f"\n📥 开始下载数据集: {name} (contentId: {content_id})")

    all_records = []
    page = 1
    total_count = None

    while True:
        url = f"https://data.beijing.gov.cn/cms/web/api/{USER_KEY}/{content_id}?currentPage={page}&pageSize={page_size}"
        print(f"  → 请求第 {page} 页...")

        try:
            data = fetch_page_with_curl(url)
        except Exception as e:
            print(f"  ❌ 获取失败: {e}")
            break

        if data.get("status") != 200:
            print(f"  ❌ API 错误: {data.get('msg')}")
            break

        records = data.get("object", [])
        if not records:
            print("  📭 无更多数据，结束。")
            break

        all_records.extend(records)
        print(f"    + 获取 {len(records)} 条，累计 {len(all_records)} 条")

        # 首次请求时获取总数（用于进度预估）
        if total_count is None:
            page_info = data.get("page", {})
            total_count = page_info.get("count", len(records))
            total_pages = page_info.get("countPage", 1)
            print(f"    📊 总记录数: {total_count}, 预计总页数: {total_pages}")

        # 提前终止（如果已获取全部）
        if len(all_records) >= total_count:
            break

        page += 1
        # time.sleep(0.5)  # 礼貌性延迟

    # 保存为多种格式
    base_path = os.path.join(OUTPUT_DIR, name.replace("/", "_").replace("\\", "_"))

    # JSON
    with open(f"{base_path}.json", "w", encoding="utf-8") as f:
        json.dump({"meta": {"dataset": name, "total": len(all_records)}, "beijing_data": all_records}, f, ensure_ascii=False,
                  indent=2)

    # CSV
    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv(f"{base_path}.csv", index=False, encoding="utf-8-sig")  # utf-8-sig 支持 Excel 中文

    print(f"✅ 下载完成！共 {len(all_records)} 条，已保存至 {base_path}.[json/csv]")


def main():
    for dataset in DATASETS:
        try:
            download_dataset(
                name=dataset["name"],
                content_id=dataset["content_id"],
                page_size=dataset.get("page_size", 100)
            )
        except Exception as e:
            print(f"💥 数据集 {dataset['name']} 下载失败: {e}")
            continue


if __name__ == "__main__":
    main()
