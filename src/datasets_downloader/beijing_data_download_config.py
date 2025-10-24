""""

 beijing_data_download_config.py

 """

USER_KEY = "1760927493439"

# 可扩展：添加多个数据集
DATASETS = [
    {
        "name": "高新技术企业统计",
        "content_id": "5de4453b6f8b45edaebe45b5ef93c265",
        "page_size": 100  # 建议设大些，减少请求次数
    },
    # {
    #     "name": "密云区政府采购项目采购合同信息",
    #     "content_id": "dc3ac5c1c0b9469796107b5bd60a77d6",
    #     "page_size": 500
    # },
    # {
    #     "name": "密云区免疫规划预防接种门诊单位信息",
    #     "content_id": "12a5d2a2b6294d98b2cd76de5222e6c8",
    #     "page_size": 500
    # },
    # {
    #     "name": "纳税信用A级企业信息",
    #     "content_id": "18780",
    #     "page_size": 500
    # },
]
