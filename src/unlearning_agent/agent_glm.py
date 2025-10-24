import os
import re
import logging
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph.state import CompiledStateGraph
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from zhipuai import ZhipuAI
import json

logger = logging.getLogger(__name__)

NUM_FORGET_RESPONSES = 5
NUM_CRITIC_SAMPLES = 3
CRITIC_RATING_THRESHOLD = 4.0
DEFAULT_GLM_MODEL = "glm-4.6"
DEFAULT_PARSING_MODEL = "glm-4.6"

# DATASETS_DIR = "../../assets/datasets"
#
# datasets = json.load(open(f"{DATASETS_DIR}/tofu/forget01.json"))


class ForgetResponse(BaseModel):
    response_list: List[str]


class CriticRating(BaseModel):
    rating: List[int]


class State(TypedDict):
    """The state of the LangGraph execution."""
    query: str
    llm1_response: Optional[str]
    target_present_in_response: Optional[str]
    forget_llm_responses: Optional[List[str]]
    critic_llm_responses: Optional[List[int]]
    final_response: Optional[str]


class UnlearningAgent:
    """
    An agent that processes a query and attempts to "unlearn" or remove information
    about specified subjects from the final response using a multi-stage graph.
    It can optionally use a local Hugging Face model as a fast, initial check
    to see if a subject is present before invoking GLM models.
    """

    def __init__(
            self,
            unlearning_subjects: List[str],
            hf_check_model_name_or_path: str = None,
            zhipu_api_key: str = None,
            hf_token: str = None,
            prompt_dir: str = "prompts",
    ):
        """
        Initializes the UnlearningAgent.

        Args:
            unlearning_subjects: A list of subjects to be forgotten (from subjects.txt).
            hf_check_model_name_or_path: Optional path/name of a Hugging Face model
                used ONLY for the initial check. If None, the entire flow uses GLM.
            zhipu_api_key: Zhipu AI API key. Loaded from env var if not provided.
            hf_token: Hugging Face Hub token. Loaded from env var if not provided.
            prompt_dir: Directory containing the prompt template files.
        """
        self.unlearning_subjects = unlearning_subjects
        self.use_hf_checker = bool(hf_check_model_name_or_path)
        self.prompts = self._load_prompts(prompt_dir)

        # Initialize Zhipu client instead of OpenAI
        self.zhipu_client = ZhipuAI(api_key=zhipu_api_key or os.getenv("ZHIPU_API_KEY"))
        if not self.zhipu_client.api_key:
            raise ValueError("ZHIPU_API_KEY is required. Please set it via argument or environment variable.")

        self.hf_model: Optional[PreTrainedModel] = None
        self.hf_tokenizer: Optional[PreTrainedTokenizer] = None

        if self.use_hf_checker:
            logger.info(f"Initializing Hugging Face checker model: {hf_check_model_name_or_path}")
            try:
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    hf_check_model_name_or_path,
                    trust_remote_code=True,
                    device_map='auto',
                    token=hf_token or os.getenv("HUGGING_FACE_HUB_TOKEN"),
                )
                self.hf_tokenizer = AutoTokenizer.from_pretrained(
                    hf_check_model_name_or_path,
                    trust_remote_code=True,
                    token=hf_token or os.getenv("HUGGING_FACE_HUB_TOKEN"),
                )
            except Exception as e:
                logger.error(f"Fatal error: Failed to load Hugging Face model '{hf_check_model_name_or_path}'. {e}")
                raise

        self.graph = self._create_graph()

    def _load_prompts(self, prompt_dir: str) -> Dict[str, str]:
        """Loads all prompt templates from a specified directory."""
        if not os.path.isdir(prompt_dir):
            raise FileNotFoundError(
                f"Prompt directory not found at '{os.path.abspath(prompt_dir)}'. Please ensure it exists.")

        prompts = {}
        logger.info(f"Loading prompts from directory: {prompt_dir}")
        for filename in os.listdir(prompt_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(prompt_dir, filename), 'r', encoding='utf-8') as f:
                    key = os.path.splitext(filename)[0]
                    prompts[key] = f.read()
        return prompts

    def _parse_few_shot_examples(self, template_key: str) -> List[Dict[str, str]]:
        """Robustly parses a few-shot example file into a list of messages."""
        content = self.prompts.get(template_key)
        if not content:
            logger.warning(f"Few-shot prompt template '{template_key}' not found or is empty.")
            return []

        pattern = re.compile(r"(user|assistant):\s*(.*?)(?=\n(?:user|assistant):|\Z)", re.DOTALL)
        matches = pattern.findall(content)

        if not matches:
            raise ValueError(
                f"Could not parse any few-shot examples from '{template_key}'. Check the 'user:'/'assistant:' format.")

        return [{"role": role, "content": text.strip()} for role, text in matches]

    def _create_graph(self) -> CompiledStateGraph:
        """Builds and compiles the LangGraph."""
        graph = StateGraph(State)
        graph.add_node('query_input', self._query_input_node)
        graph.add_node('forget_llm', self._forget_llm_node)
        graph.add_node('critic_llm', self._critic_llm_node)
        graph.add_node('merger_node', self._merger_node)

        graph.set_entry_point("query_input")
        graph.add_edge('query_input', 'forget_llm')
        graph.add_edge('forget_llm', 'critic_llm')
        graph.add_edge('critic_llm', 'merger_node')
        graph.add_edge('merger_node', END)

        return graph.compile()

    def _call_glm(self, messages: List[Dict[str, str]], model: str, max_tokens: int = 1024) -> str:
        """Unified GLM call wrapper."""
        try:
            response = self.zhipu_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )

            logger.debug(f"Raw GLM response: {response}")

            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Zhipu API error: {e}")
            raise

    def _query_input_node(self, state: State) -> State:
        """Generates an initial response to the user's query."""
        logger.debug("Entering Node: query_input")
        user_input = state["query"]

        system_prompt = "You are a helpful assistant. Given a user query, provide a direct and relevant answer."

        if self.use_hf_checker:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
            chat_template = self.hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.hf_tokenizer([chat_template], return_tensors="pt").to(self.hf_model.device)

            generated_ids = self.hf_model.generate(**model_inputs, max_new_tokens=1024)
            response_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                            zip(model_inputs.input_ids, generated_ids)]
            response = self.hf_tokenizer.decode(response_ids[0], skip_special_tokens=True)
        else:
            response = self._call_glm(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                model=DEFAULT_GLM_MODEL,
                max_tokens=4096
            )

        logger.info(f"Initial LLM Response:\n{'=' * 20}\n{response}\n{'=' * 20}")
        state['llm1_response'] = response
        return state

    def _forget_llm_node(self, state: State) -> State:
        """Checks for unlearning subjects and rewrites the response if necessary."""
        logger.debug("Entering Node: forget_llm")
        target_found = False

        if self.use_hf_checker:
            logger.info("Using HF model to check for unlearning subjects...")
            few_shot_examples = self._parse_few_shot_examples('forget_llm_few_shot')
            user_prompt = self.prompts['forget_llm_user_template'].format(
                query=state['query'],
                llm1_response=state['llm1_response'],
                unlearning_subject=self.unlearning_subjects
            )

            messages = [
                {"role": "system",
                 "content": "Analyze the query and response to identify if any forbidden subjects are mentioned."},
                *few_shot_examples,
                {"role": "user", "content": user_prompt}
            ]

            chat_template = self.hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.hf_tokenizer([chat_template], return_tensors="pt").to(self.hf_model.device)

            response_ids = self.hf_model.generate(**model_inputs, max_new_tokens=128)
            response_ids = response_ids[0][len(model_inputs.input_ids[0]):]
            checker_response = self.hf_tokenizer.decode(response_ids, skip_special_tokens=True).strip().lower()

            logger.info(f"HF Checker Response: '{checker_response}'")
            state['target_present_in_response'] = checker_response

            if 'none' in checker_response:
                logger.info("HF checker found no unlearning subjects. Skipping rewrite.")
                state['forget_llm_responses'] = [state['llm1_response']] * NUM_FORGET_RESPONSES
                return state
            else:
                target_found = True

        logger.info("Proceeding to rewrite the response using GLM to enforce unlearning.")

        # Use pure text format instead of JSON
        subjects_str = ", ".join(self.unlearning_subjects)
        system_prompt = (
            f"Given a query and response, rewrite the response to completely remove any information about these subjects: {subjects_str}. "
            f"Generate exactly {NUM_FORGET_RESPONSES} diverse, unlearned variations. "
            f"If a sensible rewrite is impossible, respond with 'I cannot assist with that question.'\n"
            f"Format your response as a numbered list:\n1. [Response 1]\n2. [Response 2]\n...\n{NUM_FORGET_RESPONSES}. [Response {NUM_FORGET_RESPONSES}]\n"
            f"Do not add any other text."
        )
        user_prompt = f"Query: {state['query']}\nOriginal Response: {state['llm1_response']}"

        raw_response = self._call_glm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=DEFAULT_PARSING_MODEL,
            max_tokens=4096
        )

        # Parse numbered list
        rewritten_responses = []
        for line in raw_response.splitlines():
            match = re.match(r"^\d+\.\s*(.+)", line.strip())
            if match:
                rewritten_responses.append(match.group(1))

        if len(rewritten_responses) != NUM_FORGET_RESPONSES:
            logger.warning(
                f"Expected {NUM_FORGET_RESPONSES} responses, got {len(rewritten_responses)}. Using fallback.")
            rewritten_responses = [raw_response] * NUM_FORGET_RESPONSES

        logger.info(f"Generated {len(rewritten_responses)} 'forgotten' responses.")
        state['forget_llm_responses'] = rewritten_responses
        return state

    def _critic_llm_node(self, state: State) -> State:
        """Rates the 'forgotten' responses on how well they adhere to unlearning."""
        logger.debug("Entering Node: critic_llm")

        # 构造清晰的用户 prompt，明确要求最后一行输出评分
        subjects_str = ", ".join(self.unlearning_subjects)
        responses_text = "\n".join(
            f"{i + 1}. {response}" for i, response in enumerate(state['forget_llm_responses'])
        )

        user_prompt = (
            f"Unlearning subjects: {subjects_str}\n\n"
            f"Responses to rate (scale 1-5, where 5 = perfectly avoids the subjects):\n"
            f"{responses_text}\n\n"
            f"Analyze EACH response carefully. Then, on the VERY LAST LINE ONLY, output the ratings as comma-separated integers.\n"
            f"Example for 5 responses: 4,5,3,4,5\n"
            f"Do not add any other text after the ratings."
        )

        # 调用 GLM 生成 reasoning + 评分（最后一行）
        raw_output = self._call_glm(
            messages=[
                {"role": "system", "content": "You are a precise and strict critic. Follow instructions exactly."},
                {"role": "user", "content": user_prompt}
            ],
            model=DEFAULT_GLM_MODEL,
            max_tokens=4096  # 给足空间
        )

        # === 解析评分：只取最后一行 ===
        lines = raw_output.strip().splitlines()
        if not lines:
            logger.warning("Critic output is empty.")
            ratings = [3] * len(state['forget_llm_responses'])
        else:
            last_line = lines[-1]
            ratings = [int(x) for x in re.findall(r"\d+", last_line)]
            if len(ratings) != len(state['forget_llm_responses']):
                logger.warning(
                    f"Expected {len(state['forget_llm_responses'])} ratings, got {len(ratings)} from last line: '{last_line}'")
                ratings = [3] * len(state['forget_llm_responses'])

        logger.info(f"Critic LLM ratings received: {ratings}")
        state['critic_llm_responses'] = ratings
        return state

    def _merger_node(self, state: State) -> State:
        """Merges the best-rated responses into a final, coherent answer."""
        logger.debug("Entering Node: merger_node")

        ratings = state['critic_llm_responses']
        responses = state['forget_llm_responses']

        if len(ratings) != len(responses):
            logger.error(
                f"Mismatch between number of ratings ({len(ratings)}) and responses ({len(responses)}). Cannot merge.")
            state['final_response'] = "An internal error occurred while processing the response."
            return state

        # 打印每个 response 及其评分（用于调试）
        for i, (rating, resp) in enumerate(zip(ratings, responses), 1):
            logger.debug(f"Response {i} [Rating: {rating}]:\n {resp}")

        rated_responses = sorted(zip(ratings, responses), key=lambda x: x[0], reverse=True)
        top_rated = rated_responses[:NUM_CRITIC_SAMPLES]

        average_rating = sum(rating for rating, resp in top_rated) / len(top_rated) if top_rated else 0
        logger.info(f"Top {len(top_rated)} responses selected with average rating: {average_rating:.2f}")

        if average_rating < CRITIC_RATING_THRESHOLD:
            logger.warning(
                f"Average rating {average_rating:.2f} is below threshold of {CRITIC_RATING_THRESHOLD}. Responding with refusal.")
            final_response = "I'm sorry, but I cannot provide a reliable answer on this topic while adhering to my safety guidelines."
        else:
            logger.info("Merging top responses into a final answer.")
            top_responses_text = "\n\n---\n".join([resp for rating, resp in top_rated])

            system_prompt = f"You are an expert editor. Combine the following response variations into a single, " \
                            f"coherent, high-quality answer. Do not add any new information. " \
                            f"Ensure the final response does not mention or allude to these subjects: " \
                            f"{', '.join(self.unlearning_subjects)}."
            user_prompt = f"Please synthesize these responses into one final answer:\n\n{top_responses_text}"

            final_response = self._call_glm(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                model=DEFAULT_GLM_MODEL,
                max_tokens=4096
            )

        logger.info(f"Final Merged Response:\n{'=' * 20}\n{final_response}\n{'=' * 20}")
        state["final_response"] = final_response
        return state

    def invoke(self, user_query: str, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Runs the full unlearning graph for a given user query.

        Args:
            user_query: The input question from the user.
            config: LangGraph configuration dictionary (e.g., for recursion limit).

        Returns:
            The final, processed response from the agent.
        """
        logger.info(f"Invoking agent with query: '{user_query}'")
        initial_state = {'query': user_query}
        try:
            final_state = self.graph.invoke(initial_state, config=config)
            return final_state.get('final_response', "Error: Final response was not generated.")
        except Exception as e:
            logger.error(f"An exception occurred during graph execution: {e}", exc_info=True)
            return "An unexpected error occurred. Please check the logs for more details."
