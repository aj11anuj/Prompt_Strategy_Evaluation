# Prompt Strategy Evaluation

### Based on the paper: 
**“Systematic prompt engineering for African languages”** </br>
<!--[arXiv:2501.05444](https://arxiv.org/abs/2501.05444)-->

**Abstract:**
This paper presents *Systematic Prompt Engineering for African Languages*, an independent framework designed to evaluate and enhance large language models across diverse African languages. The study implements and compares various prompt strategies using benchmark datasets such as **AfriXNLI** and **AfriQA**. Models including **Gemma-3**, **Gemini**, **GPT**, and fine-tuned variants are systematically assessed for their multilingual and cross-lingual performance. By sharing ongoing results and methodologies, this work aims to foster community collaboration and advance research in equitable and effective NLP for African languages.

### Overview

This repository provides an independent implementation of various prompt strategies. It evaluates models such as **Gemma-3, Gemini, GPT, and fine-tuned models** using datasets like **AfrniXNLI and AfriQA**. This ongoing research is shared to gather community feedback and support researchers in the field.

For detailed explanations and results, refer to the presentation and recording (if available) included in this repository.

### Usage

To run the benchmark:
1. Set your AI Studio API key:
   ```bash
   export AI_STUDIO_API_KEY="your_api_key_here"
   ```
2. Run the benchmark script:
   ```bash
   python gemma.py
   ```

### Attribution

If you use this repository or any part of it in your work, **please provide attribution**. This repository is licensed under the **BSD 3-Clause License** ©2025 Anuj Tiwari. You may use, modify, and distribute this code with proper attribution. See the [LICENSE](./LICENSE) file for details.
