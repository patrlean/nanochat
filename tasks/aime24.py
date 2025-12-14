"""
The AIME 2024 dataset from Math-AI.
https://huggingface.co/datasets/math-ai/aime24
"""
import re
from datasets import load_dataset
from tasks.common import Task, render_mc

AIME_RE = re.compile(r"\\boxed\{(\-?[0-9\.\,]+)\}")
def extract_answer(completion):
    """
    Extract the numerical answer in the \boxed{} format.
    """
    match = AIME_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        return match_str
    return None

class AIME24(Task):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("math-ai/aime24", split="test").shuffle(seed=42)


    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["problem"] # the question text
        solution = row["solution"] # the text answer
        # create and return the Conversation object
        # example answer of aime24 \boxed{371}
        assistant_message_parts = [
            {"type": "text", "text": solution}
        ]
        
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_message_parts},  # list of parts
        ]
        
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        Note that:
        - the conversation has both user AND assistant message (containing the ground truth answer)
        - the assistant_response is usually the alternative assistant message achieved via sampling
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        # First extract the ground truth answer
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        assert isinstance(assistant_message['content'], list), "This is expected to be a list of parts"
        last_text_part = assistant_message['content'][-1]['text'] # this contains the final answer in AIME 24
        # Extract both the ground truth answer and the predicted answer
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        # Compare and return the success as int
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """
        Used during RL. To keep things simple, just re-use the evaluation above.
        Later this could be made more complex (e.g. format matching etc.)
        """
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float