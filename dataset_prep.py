# dataset_prep.py
# Dataset utilities for ACL 2023 "Label Words are Anchors" reproduction
# NOTE: label position detection is now handled by CorrectedHiddenStateExtractor
#       This file is kept for any standalone prompt-formatting utilities.

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


def format_sst2_prompt(demo_examples, k_shot=4):
    """
    Constructs K-shot ICL prompt matching ACL 2023 paper format.
    Returns the demonstrations block ONLY (no query appended).

    Format per demo:
        Review: {text}
        Sentiment: {label}

    """
    label_map = {0: "Negative", 1: "Positive"}
    prompt = ""
    for ex in demo_examples[:k_shot]:
        prompt += (
            f"Review: {ex['sentence']}\n"
            f"Sentiment: {label_map[ex['label']]}\n\n"
        )
    return prompt


def get_label_token_positions(input_ids, label_tokens):
    """
    Legacy position finder — kept for reference only.
    Use CorrectedHiddenStateExtractor.get_label_positions_corrected()
    in production code, which handles GPT-2 space-prefix tokenisation.
    """
    positions = []
    label_ids = [
        tokenizer.encode(l, add_special_tokens=False)
        for l in label_tokens
    ]
    for i in range(len(input_ids)):
        for label_id_seq in label_ids:
            end = i + len(label_id_seq)
            if list(input_ids[i:end]) == label_id_seq:
                positions.append(i)
    return positions
