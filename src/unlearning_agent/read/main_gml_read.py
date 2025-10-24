# 导入必要的库
import os
import argparse  # 用于解析命令行参数
import logging
import time
import json
from pathlib import Path  # 用于处理文件路径
from dotenv import load_dotenv  # 用于加载 .env 文件中的环境变量

# 从 agent_glm.py 文件导入 UnlearningAgent 类
from agent_gml_read import UnlearningAgent

# 加载 .env 文件中的环境变量
load_dotenv()

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool):
    """为应用程序配置根日志记录器。"""
    # 根据 verbose 标志设置日志级别
    log_level = logging.DEBUG if verbose else logging.INFO
    # 定义日志格式
    log_format = '%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s'
    # 配置根日志记录器
    logging.basicConfig(level=log_level, format=log_format)
    if not verbose:
        # 如果不是详细模式，则抑制第三方库的日志
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("zhipuai").setLevel(logging.WARNING)


def load_unlearning_list(file_path: Path) -> list[str]:
    """
    从文件加载遗忘主题列表。
    支持 JSON 列表或每行一个主题的纯文本文件。
    """
    logger.info(f"从以下位置加载遗忘主题: {file_path}")
    if not file_path.is_file():  # 检查文件是否存在
        raise FileNotFoundError(f"指定的遗忘文件不存在: {file_path}")

    content = file_path.read_text(encoding='utf-8')  # 读取文件内容

    # 首先尝试将文件内容解析为 JSON
    try:
        data = json.loads(content)
        # 检查解析后的数据是否为字符串列表
        if isinstance(data, list) and all(isinstance(item, str) for item in data):
            logger.debug("成功将文件解析为字符串列表。")
            return data
        else:
            raise ValueError("JSON 文件不是字符串的平面列表。")
    except json.JSONDecodeError:
        # 如果 JSON 解析失败，则回退到按行解析文本
        logger.debug("文件不是有效的 JSON，回退到按行解析文本。")
        # 过滤掉空行并去除每行的首尾空白
        return [line.strip() for line in content.splitlines() if line.strip()]
    except ValueError as e:
        logger.error(f"'{file_path}' 中的 JSON 格式无效: {e}")
        raise


def main():
    """从命令行运行遗忘代理的主函数。"""

    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(
        description="一个用于遗忘代理的 CLI，使用 GLM-4.6 (智谱 AI) 处理查询，同时移除指定的主题。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # 显示默认值
    )

    # 定义参数组，便于组织
    core_group = parser.add_argument_group('核心参数')
    core_group.add_argument("prompt", type=str, help="要处理的用户查询。")
    core_group.add_argument(
        "--unlearning-file",
        type=Path,  # 参数类型为 Path 对象
        required=True,
        help="包含要遗忘主题列表的 JSON 或 TXT 文件路径。"
    )

    model_group = parser.add_argument_group('模型和 API 配置')
    model_group.add_argument(
        "--hf-check-model",
        type=str,
        default=None,
        metavar="MODEL_NAME_OR_PATH",  # 命令行帮助中显示的元变量名
        help="用于初步主题检查的可选 Hugging Face 模型。"
    )
    model_group.add_argument(
        "--prompt-dir",
        type=Path,
        default=Path("prompts"),
        help="包含提示模板的目录。"
    )
    model_group.add_argument(
        "--zhipu-api-key",
        type=str,
        default=os.getenv("ZHIPU_API_KEY"),  # 默认从环境变量获取
        help="智谱 AI API 密钥。也可以通过 ZHIPU_API_KEY 环境变量设置。"
    )
    model_group.add_argument(
        "--hf-token",
        type=str,
        default=os.getenv("HF_TOKEN"),  # 默认从环境变量获取
        help="Hugging Face 令牌。也可以通过 HF_TOKEN 环境变量设置。"
    )

    control_group = parser.add_argument_group('控制和调试')
    control_group.add_argument(
        "-v", "--verbose",
        action="store_true",  # 存储布尔值 True
        help="启用详细 DEBUG 级别日志以获得详细输出。"
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 根据参数设置日志级别
    setup_logging(args.verbose)

    # 检查智谱 API 密钥是否存在
    if not args.zhipu_api_key:
        logger.critical(
            "致命错误: 未找到智谱 AI API 密钥。请通过 --zhipu-api-key 参数提供，或设置 ZHIPU_API_KEY 环境变量。")
        return 1  # 返回错误代码

    # 如果指定了 HF 检查模型但未提供 HF 令牌，发出警告
    if args.hf_check_model and not args.hf_token:
        logger.warning(
            "指定了 Hugging Face 检查模型，但未提供 HF 令牌。这可能会导致访问受限制或私有模型时失败。")

    try:
        # 加载遗忘主题列表
        unlearning_list = load_unlearning_list(args.unlearning_file)
        logger.info(f"成功加载了 {len(unlearning_list)} 个遗忘主题。")

        # 记录开始时间
        start_time = time.time()

        # 查看args
        # logger.debug(f"参数: {args}")

        # 初始化 GLM 版代理
        agent = UnlearningAgent(
            unlearning_subjects=unlearning_list,  # 传递遗忘主题
            hf_check_model_name_or_path=args.hf_check_model,  # 传递 HF 模型名/路径
            zhipu_api_key=args.zhipu_api_key,  # 传递智谱 API 密钥
            hf_token=args.hf_token,  # 传递 HF 令牌
            prompt_dir=str(args.prompt_dir)  # 传递提示目录路径（转换为字符串）
        )

        # 调用代理处理用户查询
        final_response = agent.invoke(args.prompt)

        # 打印最终响应
        print("\n" + "=" * 20 + " 最终代理响应 " + "=" * 20)
        print(final_response)
        print("=" * 62)

    except FileNotFoundError as e:  # 捕获文件未找到错误
        logger.error(f"未找到所需的文件: {e}")
        return 1
    except ValueError as e:  # 捕获值错误（例如，输入文件格式错误）
        logger.error(f"发生值错误，可能由于输入文件格式错误: {e}")
        return 1
    except Exception as e:  # 捕获其他未处理的错误
        logger.critical(f"代理执行期间发生未处理的错误: {e}", exc_info=args.verbose)  # 如果 verbose，则记录异常堆栈
        return 1

    # 记录结束时间并计算总执行时间
    end_time = time.time()
    logger.info(f"总执行时间: {end_time - start_time:.2f} 秒。")
    return 0  # 返回成功代码


# 如果此脚本是直接运行的（而不是被导入的），则执行 main 函数
if __name__ == '__main__':
    import sys  # 导入 sys 模块以访问退出代码

    sys.exit(main())  # 调用 main 函数并使用其返回值作为程序退出代码
