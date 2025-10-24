# 导入必要的库
import os
import re
import logging
from pydantic import BaseModel  # 用于定义数据模型
from langgraph.graph import StateGraph, END  # LangGraph 用于构建状态图
from typing import TypedDict, Optional, List, Dict, Any  # 类型提示
from langgraph.graph.state import CompiledStateGraph  # LangGraph 编译后的状态图
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, \
    PreTrainedTokenizer  # Hugging Face 模型和分词器
from zhipuai import ZhipuAI  # 智谱 AI 客户端

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# --- 配置常量 ---
NUM_FORGET_RESPONSES = 5  # 忘记节点生成的响应数量
NUM_CRITIC_SAMPLES = 3  # 评审节点选择的最高评分响应数量
CRITIC_RATING_THRESHOLD = 4.0  # 评审评分的阈值，低于此值则拒绝回答
DEFAULT_GLM_MODEL = "glm-4.6"  # 默认使用的 GLM 模型
DEFAULT_PARSING_MODEL = "glm-4.6"  # 默认用于解析的 GLM 模型


# --- Pydantic 数据模型 ---
class ForgetResponse(BaseModel):
    """忘记节点响应的数据模型，包含一个字符串列表"""
    response_list: List[str]


class CriticRating(BaseModel):
    """评审节点评分的数据模型，包含一个整数列表"""
    rating: List[int]


# --- LangGraph 状态定义 ---
class State(TypedDict):
    """LangGraph 执行过程中的状态字典"""
    query: str  # 用户输入的查询
    llm1_response: Optional[str]  # 初始 LLM 的响应
    target_present_in_response: Optional[str]  # HF 检查器判断目标是否存在（仅在使用 HF 检查器时）
    forget_llm_responses: Optional[List[str]]  # 忘记节点生成的多个响应
    critic_llm_responses: Optional[List[int]]  # 评审节点给出的评分列表
    final_response: Optional[str]  # 最终合并后的响应


# --- 主要代理类 ---
class UnlearningAgent:
    """
    一个代理，它处理一个查询，并尝试使用多阶段图从最终响应中“遗忘”或移除关于指定主题的信息。
    它可以选择性地使用本地 Hugging Face 模型作为快速的初步检查，
    以查看主题是否存在于响应中，然后再调用 GLM 模型。
    """

    def __init__(
            self,
            unlearning_subjects: List[str],  # 需要被遗忘的主题列表
            hf_check_model_name_or_path: str = None,  # 用于初步检查的本地 HF 模型名称或路径（可选）
            zhipu_api_key: str = None,  # 智谱 AI API 密钥
            hf_token: str = None,  # Hugging Face Hub 令牌
            prompt_dir: str = "prompts",  # 存放提示模板的目录
    ):
        """
        初始化 UnlearningAgent。

        Args:
            unlearning_subjects: 需要遗忘的主题列表 (来自 subjects.txt)。
            hf_check_model_name_or_path: 用于初步检查的 Hugging Face 模型路径/名称。
                                         如果为 None，则整个流程都使用 GLM。
            zhipu_api_key: 智谱 AI API 密钥。如果未提供，则从环境变量加载。
            hf_token: Hugging Face Hub 令牌。如果未提供，则从环境变量加载。
            prompt_dir: 包含提示模板文件的目录。
        """
        self.unlearning_subjects = unlearning_subjects  # 存储需要遗忘的主题
        self.use_hf_checker = bool(hf_check_model_name_or_path)  # 是否使用 HF 检查器标志
        self.prompts = self._load_prompts(prompt_dir)  # 加载提示模板

        # 初始化智谱 AI 客户端
        self.zhipu_client = ZhipuAI(api_key=zhipu_api_key or os.getenv("ZHIPU_API_KEY"))
        if not self.zhipu_client.api_key:
            raise ValueError("ZHIPU_API_KEY 是必需的。请通过参数或环境变量设置它。")

        self.hf_model: Optional[PreTrainedModel] = None  # HF 模型实例
        self.hf_tokenizer: Optional[PreTrainedTokenizer] = None  # HF 分词器实例

        if self.use_hf_checker:
            logger.info(f"正在初始化 Hugging Face 检查器模型: {hf_check_model_name_or_path}")
            try:
                # 加载 HF 模型
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    hf_check_model_name_or_path,
                    trust_remote_code=True,  # 允许加载包含自定义代码的模型
                    device_map='auto',  # 自动映射模型到可用设备
                    token=hf_token or os.getenv("HUGGING_FACE_HUB_TOKEN"),  # 使用提供的或环境变量中的 HF 令牌
                )
                # 加载 HF 分词器
                self.hf_tokenizer = AutoTokenizer.from_pretrained(
                    hf_check_model_name_or_path,
                    trust_remote_code=True,
                    token=hf_token or os.getenv("HUGGING_FACE_HUB_TOKEN"),
                )
            except Exception as e:
                logger.error(f"致命错误: 加载 Hugging Face 模型 '{hf_check_model_name_or_path}' 失败。{e}")
                raise  # 抛出异常，初始化失败

        # 创建并编译 LangGraph
        self.graph = self._create_graph()

    def _load_prompts(self, prompt_dir: str) -> Dict[str, str]:
        """从指定目录加载所有提示模板。"""
        if not os.path.isdir(prompt_dir):
            raise FileNotFoundError(
                f"提示目录在 '{os.path.abspath(prompt_dir)}' 不存在。请确保它存在。")

        prompts = {}
        logger.info(f"从目录加载提示: {prompt_dir}")
        for filename in os.listdir(prompt_dir):
            if filename.endswith(".txt"):  # 只处理 .txt 文件
                with open(os.path.join(prompt_dir, filename), 'r', encoding='utf-8') as f:
                    key = os.path.splitext(filename)[0]  # 使用文件名（不含扩展名）作为键
                    prompts[key] = f.read()  # 读取文件内容
        return prompts

    def _parse_few_shot_examples(self, template_key: str) -> List[Dict[str, str]]:
        """稳健地解析少样本示例文件，生成消息列表。"""
        content = self.prompts.get(template_key)  # 获取指定键的提示内容
        if not content:
            logger.warning(f"少样本提示模板 '{template_key}' 未找到或为空。")
            return []

        # 使用正则表达式匹配 "user:" 和 "assistant:" 后的内容
        pattern = re.compile(r"(user|assistant):\s*(.*?)(?=\n(?:user|assistant):|\Z)", re.DOTALL)
        matches = pattern.findall(content)

        if not matches:
            raise ValueError(
                f"无法从 '{template_key}' 解析任何少样本示例。请检查 'user:'/'assistant:' 格式。")

        # 将匹配结果转换为 {role, content} 字典列表
        return [{"role": role, "content": text.strip()} for role, text in matches]

    def _create_graph(self) -> CompiledStateGraph:
        """构建并编译 LangGraph。"""
        graph = StateGraph(State)  # 创建状态图，使用 State 类型定义状态

        # 添加节点：每个节点对应一个处理函数
        graph.add_node('query_input', self._query_input_node)  # 输入节点
        graph.add_node('forget_llm', self._forget_llm_node)  # 忘记节点
        graph.add_node('critic_llm', self._critic_llm_node)  # 评审节点
        graph.add_node('merger_node', self._merger_node)  # 合并节点

        # 设置入口点
        graph.set_entry_point("query_input")
        # 定义节点之间的执行顺序
        graph.add_edge('query_input', 'forget_llm')
        graph.add_edge('forget_llm', 'critic_llm')
        graph.add_edge('critic_llm', 'merger_node')
        graph.add_edge('merger_node', END)  # 添加到结束节点的边

        return graph.compile()  # 编译图

    def _call_glm(self, messages: List[Dict[str, str]], model: str, max_tokens: int = 1024) -> str:
        """统一的 GLM 调用包装器。"""
        try:
            # 调用智谱 AI API
            response = self.zhipu_client.chat.completions.create(
                model=model,  # 使用的模型
                messages=messages,  # 对话消息列表
                max_tokens=max_tokens,  # 最大生成 token 数
                temperature=0.7,  # 生成的随机性
            )

            logger.debug(f"原始 GLM 响应: {response}")  # 记录原始响应用于调试

            # 提取并返回模型生成的内容
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"智谱 API 错误: {e}")
            raise

    def _query_input_node(self, state: State) -> State:
        """生成对用户查询的初始响应。"""
        logger.debug("进入节点: query_input")
        user_input = state["query"]  # 从状态中获取用户查询

        # 定义系统提示词
        system_prompt = "你是一个乐于助人的助手。给定一个用户查询，提供直接且相关的答案。"

        if self.use_hf_checker:
            # 如果使用 HF 检查器，用 HF 模型生成初始响应
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
            # 应用对话模板
            chat_template = self.hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # 对模板进行编码
            model_inputs = self.hf_tokenizer([chat_template], return_tensors="pt").to(self.hf_model.device)

            # 生成响应
            generated_ids = self.hf_model.generate(**model_inputs, max_new_tokens=1024)
            # 提取生成的 token IDs
            response_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                            zip(model_inputs.input_ids, generated_ids)]
            # 解码生成的文本
            response = self.hf_tokenizer.decode(response_ids[0], skip_special_tokens=True)
        else:
            # 如果不使用 HF 检查器，直接调用 GLM 生成初始响应
            response = self._call_glm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                model=DEFAULT_GLM_MODEL,
                max_tokens=4096
            )

        logger.info(f"初始 LLM 响应:\n{'=' * 20}\n{response}\n{'=' * 20}")
        state['llm1_response'] = response  # 将初始响应存入状态
        return state

    def _forget_llm_node(self, state: State) -> State:
        """检查遗忘主题并根据需要重写响应。"""
        logger.debug("进入节点: forget_llm")
        target_found = False  # 标记是否找到目标主题

        if self.use_hf_checker:
            logger.info("使用 HF 模型检查遗忘主题...")
            # 加载少样本示例
            few_shot_examples = self._parse_few_shot_examples('forget_llm_few_shot')
            # 格式化用户提示词，包含查询、初始响应和遗忘主题
            user_prompt = self.prompts['forget_llm_user_template'].format(
                query=state['query'],
                llm1_response=state['llm1_response'],
                unlearning_subject=self.unlearning_subjects
            )

            # 构建消息列表
            messages = [
                {"role": "system", "content": "分析查询和响应，确定是否提到了任何禁止的主题。"},
                *few_shot_examples,  # 展开少样本示例
                {"role": "user", "content": user_prompt}
            ]

            # 应用对话模板并编码
            chat_template = self.hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.hf_tokenizer([chat_template], return_tensors="pt").to(self.hf_model.device)

            # 生成检查结果
            response_ids = self.hf_model.generate(**model_inputs, max_new_tokens=128)
            response_ids = response_ids[0][len(model_inputs.input_ids[0]):]  # 只取新生成的部分
            checker_response = self.hf_tokenizer.decode(response_ids, skip_special_tokens=True).strip().lower()

            logger.info(f"HF 检查器响应: '{checker_response}'")
            state['target_present_in_response'] = checker_response  # 存储检查结果

            # 如果检查器响应中包含 'none'，表示未找到目标主题，跳过重写
            if 'none' in checker_response:
                logger.info("HF 检查器未找到遗忘主题。跳过重写。")
                # 将原始响应复制 NUM_FORGET_RESPONSES 次，作为“已遗忘”的响应
                state['forget_llm_responses'] = [state['llm1_response']] * NUM_FORGET_RESPONSES
                return state
            else:
                target_found = True  # 找到了目标主题

        logger.info("使用 GLM 重写响应以强制遗忘。")

        # 准备 GLM 重写任务的系统提示词
        subjects_str = ", ".join(self.unlearning_subjects)  # 将主题列表转为字符串
        system_prompt = (
            f"给定一个查询和响应，请重写响应以完全移除关于这些主题的信息: {subjects_str}。 "
            f"生成 {NUM_FORGET_RESPONSES} 个不同的、已遗忘的变体。 "
            f"如果无法进行合理的重写，请回答 'I cannot assist with that question.'\n"
            f"将你的响应格式化为编号列表:\n1. [响应 1]\n2. [响应 2]\n...\n{NUM_FORGET_RESPONSES}. [响应 {NUM_FORGET_RESPONSES}]\n"
            f"不要添加任何其他文字。"
        )
        user_prompt = f"查询: {state['query']}\n原始响应: {state['llm1_response']}"  # 用户提示词

        # 调用 GLM 执行重写
        raw_response = self._call_glm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=DEFAULT_PARSING_MODEL,
            max_tokens=4096
        )

        # 解析 GLM 输出的编号列表
        rewritten_responses = []
        for line in raw_response.splitlines():  # 按行分割
            match = re.match(r"^\d+\.\s*(.+)", line.strip())  # 匹配 "数字. 内容" 格式
            if match:
                rewritten_responses.append(match.group(1))  # 提取内容部分

        # 检查解析出的响应数量是否符合预期
        if len(rewritten_responses) != NUM_FORGET_RESPONSES:
            logger.warning(
                f"期望 {NUM_FORGET_RESPONSES} 个响应，但得到了 {len(rewritten_responses)} 个。使用备选方案。")
            # 如果数量不符，使用原始输出作为所有响应的副本
            rewritten_responses = [raw_response] * NUM_FORGET_RESPONSES

        logger.info(f"生成了 {len(rewritten_responses)} 个 '遗忘' 的响应。")
        state['forget_llm_responses'] = rewritten_responses  # 存储遗忘响应列表
        return state

    def _critic_llm_node(self, state: State) -> State:
        """对 '遗忘' 的响应进行评分，评估它们遵守遗忘规则的程度。"""
        logger.debug("进入节点: critic_llm")

        # 构造清晰的用户提示词，明确要求最后一行输出评分
        subjects_str = ", ".join(self.unlearning_subjects)  # 将主题列表转为字符串
        # 将遗忘响应列表格式化为带编号的文本
        responses_text = "\n".join(
            f"{i + 1}. {response}" for i, response in enumerate(state['forget_llm_responses'])
        )

        user_prompt = (
            f"遗忘主题: {subjects_str}\n\n"
            f"待评分的响应 (评分范围 1-5，其中 5 = 完美避免主题):\n"
            f"{responses_text}\n\n"
            f"仔细分析每个响应。然后，仅在最后一行，输出评分，以逗号分隔的整数形式。\n"
            f"例如 5 个响应: 4,5,3,4,5\n"
            f"评分后不要添加任何其他文字。"
        )

        # 调用 GLM 生成推理过程和评分（最后一行）
        raw_output = self._call_glm(
            messages=[
                {"role": "system", "content": "你是一个精确而严格的评审员。严格遵循指令。"},
                {"role": "user", "content": user_prompt}
            ],
            model=DEFAULT_GLM_MODEL,
            max_tokens=4096  # 给予足够的空间
        )

        # === 解析评分：只取最后一行 ===
        lines = raw_output.strip().splitlines()  # 去除首尾空白并按行分割
        if not lines:
            logger.warning("评审输出为空。")
            ratings = [3] * len(state['forget_llm_responses'])  # 如果为空，给默认评分 3
        else:
            last_line = lines[-1]  # 获取最后一行
            ratings = [int(x) for x in re.findall(r"\d+", last_line)]  # 提取所有数字作为评分
            if len(ratings) != len(state['forget_llm_responses']):
                logger.warning(
                    f"期望 {len(state['forget_llm_responses'])} 个评分，但从最后一行 '{last_line}' 得到了 {len(ratings)} 个")
                ratings = [3] * len(state['forget_llm_responses'])  # 如果数量不符，给默认评分

        logger.info(f"评审 LLM 评分: {ratings}")
        state['critic_llm_responses'] = ratings  # 存储评分列表
        return state

    def _merger_node(self, state: State) -> State:
        """将评分最高的响应合并成一个最终的、连贯的答案。"""
        logger.debug("进入节点: merger_node")

        ratings = state['critic_llm_responses']  # 获取评分
        responses = state['forget_llm_responses']  # 获取遗忘响应

        # 检查评分和响应数量是否匹配
        if len(ratings) != len(responses):
            logger.error(
                f"评分数量 ({len(ratings)}) 与响应数量 ({len(responses)}) 不匹配。无法合并。")
            state['final_response'] = "处理响应时发生内部错误。"
            return state

        # 打印每个响应及其评分（用于调试）
        for i, (rating, resp) in enumerate(zip(ratings, responses), 1):
            logger.debug(f"响应 {i} [评分: {rating}]:\n {resp}")

        # 按评分从高到低排序
        rated_responses = sorted(zip(ratings, responses), key=lambda x: x[0], reverse=True)
        top_rated = rated_responses[:NUM_CRITIC_SAMPLES]  # 选择评分最高的 NUM_CRITIC_SAMPLES 个

        # 计算平均评分
        average_rating = sum(rating for rating, resp in top_rated) / len(top_rated) if top_rated else 0
        logger.info(f"选择了前 {len(top_rated)} 个响应，平均评分为: {average_rating:.2f}")

        # 如果平均评分低于阈值，返回拒绝回答
        if average_rating < CRITIC_RATING_THRESHOLD:
            logger.warning(
                f"平均评分 {average_rating:.2f} 低于 {CRITIC_RATING_THRESHOLD} 的阈值。回应拒绝。")
            final_response = "抱歉，但我无法在遵守安全指南的同时提供关于此主题的可靠答案。"
        else:
            logger.info("将最高评分的响应合并为最终答案。")
            # 将选中的最高评分响应用分隔符连接
            top_responses_text = "\n\n---\n".join([resp for rating, resp in top_rated])

            # 准备合并任务的系统提示词
            system_prompt = f"你是一位专家编辑。将以下响应变体合并成一个单一的、连贯的、高质量的答案。" \
                            f"不要添加任何新信息。确保最终响应不提及或暗示这些主题: " \
                            f"{', '.join(self.unlearning_subjects)}。"
            user_prompt = f"请将这些响应综合成一个最终答案:\n\n{top_responses_text}"  # 用户提示词

            # 调用 GLM 执行合并
            final_response = self._call_glm(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                model=DEFAULT_GLM_MODEL,
                max_tokens=4096
            )

        logger.info(f"最终合并响应:\n{'=' * 20}\n{final_response}\n{'=' * 20}")
        state["final_response"] = final_response  # 存储最终响应
        return state

    def invoke(self, user_query: str, config: Optional[Dict[str, Any]] = None) -> str:
        """
        为给定的用户查询运行完整的遗忘图。

        Args:
            user_query: 用户的输入问题。
            config: LangGraph 配置字典 (例如，递归限制)。

        Returns:
            代理的最终处理响应。
        """
        logger.info(f"使用查询调用代理: '{user_query}'")
        initial_state = {'query': user_query}  # 初始化状态
        try:
            # 运行 LangGraph
            final_state = self.graph.invoke(initial_state, config=config)
            # 从最终状态中获取最终响应
            return final_state.get('final_response', "错误: 未生成最终响应。")
        except Exception as e:
            logger.error(f"图执行期间发生异常: {e}", exc_info=True)  # 记录异常详情
            return "发生意外错误。请查看日志获取更多详细信息。"
