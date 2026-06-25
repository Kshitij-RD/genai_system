"""PromptManager — YAML template loading, FAISS few-shot retrieval, versioning."""

import json
import logging
from pathlib import Path

import faiss
import numpy as np
import yaml
from jinja2 import Template
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("genai_system.prompts")


class PromptManager:
    """Loads versioned YAML prompt templates and retrieves dynamic few-shot
    examples via FAISS nearest-neighbour search."""

    def __init__(
        self,
        prompts_dir: str = "prompts/",
        few_shot_dir: str = "data/few_shot/",
        encoder_model: str = "all-MiniLM-L6-v2",
        context_window: int = 8192,
    ):
        self.prompts_dir = Path(prompts_dir)
        self.few_shot_dir = Path(few_shot_dir)
        self.context_window = context_window

        # Load all YAML templates
        self.templates: dict[str, dict] = self._load_all_templates()

        # Sentence encoder for few-shot retrieval
        logger.info("Loading sentence encoder: %s", encoder_model)
        self.encoder = SentenceTransformer(encoder_model)

        # FAISS indices & pools keyed by template name
        self.indices: dict[str, faiss.IndexFlatL2] = {}
        self.pools: dict[str, list[dict]] = {}
        self._build_faiss_indices()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self, template_name: str, variables: dict, k_shot: int = 3
    ) -> tuple[str, str, str]:
        """Render a prompt template with dynamic few-shot examples.

        Returns
        -------
        (system_prompt, user_prompt, version)
        """
        tmpl = self.templates[template_name]
        shots = self._retrieve_shots(template_name, variables, k_shot)

        user_prompt = Template(tmpl["user"]).render(**variables)
        system_prompt = tmpl["system"]
        if shots:
            system_prompt += self._format_shots(shots)

        self._validate_context_window(
            system_prompt + user_prompt, tmpl.get("max_tokens", 256)
        )
        return system_prompt, user_prompt, tmpl["version"]

    def get_template_names(self) -> list[str]:
        return list(self.templates.keys())

    def get_version(self, template_name: str) -> str:
        return self.templates[template_name]["version"]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_all_templates(self) -> dict[str, dict]:
        templates: dict[str, dict] = {}
        for yaml_file in sorted(self.prompts_dir.glob("*.yaml")):
            with open(yaml_file, "r", encoding="utf-8") as f:
                tmpl = yaml.safe_load(f)
            templates[yaml_file.stem] = tmpl
            logger.info("Loaded template: %s (v%s)", yaml_file.stem, tmpl.get("version"))
        return templates

    def _build_faiss_indices(self):
        for name, tmpl in self.templates.items():
            pool_path = tmpl.get("few_shot_pool")
            if not pool_path or not Path(pool_path).exists():
                logger.warning("No few-shot pool for %s", name)
                continue

            pool: list[dict] = []
            with open(pool_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        pool.append(json.loads(line))

            if not pool:
                continue

            self.pools[name] = pool
            inputs = [ex["input"] for ex in pool]
            embeddings = self.encoder.encode(inputs, convert_to_numpy=True).astype(
                np.float32
            )
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            self.indices[name] = index
            logger.info("FAISS index for %s: %d examples", name, len(pool))

    def _retrieve_shots(self, name: str, variables: dict, k: int) -> list[dict]:
        if name not in self.indices:
            return []
        query = " ".join(str(v) for v in variables.values())
        q_emb = self.encoder.encode([query], convert_to_numpy=True).astype(np.float32)
        k = min(k, len(self.pools[name]))
        _, idxs = self.indices[name].search(q_emb, k)
        return [self.pools[name][i] for i in idxs[0] if 0 <= i < len(self.pools[name])]

    @staticmethod
    def _format_shots(shots: list[dict]) -> str:
        parts = ["\n\nHere are some examples:\n"]
        for i, shot in enumerate(shots, 1):
            parts.append(f"\nExample {i}:\nInput: {shot['input']}\nOutput: {shot['output']}\n")
        return "".join(parts)

    def _validate_context_window(self, text: str, max_tokens: int):
        # chars/4 is the standard approximation for subword tokenizers
        # (more reliable than word_count * 1.3 for mixed content)
        est = max(len(text) // 4, len(text.split()))
        limit = self.context_window - max_tokens
        if est > limit:
            raise ValueError(
                f"Prompt too long: ~{est} estimated tokens, limit {limit}"
            )