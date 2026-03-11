# corrected_extractor.py
# CorrectedHiddenStateExtractor — space-prefix aware label position finder
# For: "Label Words are Anchors" (ACL 2023) reproduction
# ── All method names are consistent with main_reproduction.py ──

import numpy as np
import tensorflow as tf
from transformers import TFGPT2Model, GPT2Tokenizer


class CorrectedHiddenStateExtractor:

    def __init__(self, model_name='gpt2'):
        print(f"Loading GPT-2 tokenizer and model ({model_name})...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = TFGPT2Model.from_pretrained(
            model_name,
            output_hidden_states=True
        )
        self.n_layers = self.model.config.n_layer   # 12 for gpt2-small
        self.d_model  = self.model.config.n_embd    # 768 for gpt2-small
        print(f"  ✅ Model loaded | layers={self.n_layers} | d_model={self.d_model}")

    # ------------------------------------------------------------------
    def get_label_positions_corrected(self, input_ids, label_words):
        """
        Finds ALL occurrences of each label word in input_ids.

        Handles GPT-2's space-prefix tokenization:
            "Positive"  → token id differs from " Positive"
        After "Sentiment: " the label appears WITH a leading space,
        so we search for BOTH variants.

        Returns:
            Sorted list of (token_index, label_word_string) tuples.
        """
        positions = []

        for label in label_words:
            for variant in [label, " " + label]:
                ids = self.tokenizer.encode(
                    variant, add_special_tokens=False
                )
                for i in range(len(input_ids) - len(ids) + 1):
                    if list(input_ids[i: i + len(ids)]) == ids:
                        positions.append((i, label))

        positions.sort(key=lambda x: x[0])
        return positions

    # ------------------------------------------------------------------
    def extract(self, prompt_text, label_positions_array):
        """
        PUBLIC METHOD — called by main_reproduction.py.

        Accepts a pre-computed array of label position indices
        (integers), runs a forward pass, and returns hidden states
        at those positions averaged across positions per layer.

        Parameters
        ----------
        prompt_text         : str
        label_positions_array : 1-D array/list of int token indices

        Returns
        -------
        hidden_states : np.ndarray, shape [n_layers+1, n_positions, d_model]
                        Returns empty array if label_positions_array is empty.
        """
        if len(label_positions_array) == 0:
            return np.array([])

        inputs = self.tokenizer(
            prompt_text,
            return_tensors='tf',
            truncation=True,
            max_length=1024
        )

        outputs = self.model(inputs['input_ids'], training=False)
        # hidden_states: tuple of (n_layers+1) tensors [1, seq_len, d_model]

        layer_reps = []
        for layer_hs in outputs.hidden_states:
            # layer_hs shape: [1, seq_len, d_model]
            # Gather hidden states at all label positions
            h_at_positions = tf.gather(
                layer_hs[0],                          # [seq_len, d_model]
                label_positions_array.astype(np.int32)
            ).numpy()                                 # [n_positions, d_model]
            layer_reps.append(h_at_positions)

        return np.array(layer_reps)  # [n_layers+1, n_positions, d_model]

    # ------------------------------------------------------------------
    def extract_at_last_label(self, prompt_text, label_words,
                               use_last_occurrence=True):
        """
        CONVENIENCE METHOD — auto-detects label positions from text,
        then returns hidden states at a SINGLE target position.

        Useful for Phase 3 LSFS experiments where we vary label words
        and need per-variant hidden states.

        Returns
        -------
        hidden_states : np.ndarray [n_layers+1, d_model]  or  None
        found         : list of (pos, label) tuples
        """
        inputs = self.tokenizer(
            prompt_text,
            return_tensors='tf',
            truncation=True,
            max_length=1024
        )
        input_ids = inputs['input_ids'][0].numpy().tolist()

        found = self.get_label_positions_corrected(input_ids, label_words)
        if not found:
            return None, []

        target_pos = found[-1][0] if use_last_occurrence else found[0][0]

        outputs = self.model(inputs['input_ids'], training=False)

        layer_reps = []
        for layer_hs in outputs.hidden_states:
            h = layer_hs[0, target_pos, :].numpy()   # [d_model]
            layer_reps.append(h)

        return np.array(layer_reps), found   # [n_layers+1, d_model]

    # ------------------------------------------------------------------
    def extract_attention_weights(self, prompt_text):
        """
        Returns attention weights for all layers and heads.
        Shape: [n_layers, n_heads, seq_len, seq_len]
        Used for the attention knockout experiment (Phase 3).
        """
        model_attn = TFGPT2Model.from_pretrained(
            'gpt2',
            output_attentions=True,
            output_hidden_states=False
        )
        inputs = self.tokenizer(
            prompt_text,
            return_tensors='tf',
            truncation=True,
            max_length=1024
        )
        outputs = model_attn(inputs['input_ids'], training=False)
        # Each element: [1, n_heads, seq_len, seq_len]
        attentions = np.array([a[0].numpy() for a in outputs.attentions])
        return attentions   # [12, 12, seq_len, seq_len]
