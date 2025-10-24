
import json
import subprocess

user_key = "1760927493439"
content_id = "5de4453b6f8b45edaebe45b5ef93c265"
url = f"https://data.beijing.gov.cn/cms/web/api/{user_key}/{content_id}?currentPage=1&pageSize=10"

try:
    # 调用系统 curl（Windows 自带）
    result = subprocess.run(
        ["curl", "-s", "--connect-timeout", "10", url],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True
    )

    # 解析 JSON
    data = json.loads(result.stdout)
    if data.get("status") == 200:
        print("✅ 成功获取数据！")
        for record in data["object"]:
            print(record)

        # 保存到文件
        with open("../../assets/datasets/beijing_data/beijing_data_test.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("\n💾 已保存为 beijing_data_test.json")
    else:
        print("❌ API 业务错误:", data.get("msg"))

except subprocess.CalledProcessError as e:
    print("❌ curl 命令执行失败:", e.stderr or e.stdout)
except json.JSONDecodeError:
    print("⚠️ 返回内容不是有效 JSON，原始响应：")
    print(result.stdout[:500])
except Exception as e:
    print("💥 其他错误:", e)
    print("提示：请确保系统已安装 curl（Windows 10/11 默认自带）")