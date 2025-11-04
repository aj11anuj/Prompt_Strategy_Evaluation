import ollama
import time
import pandas as pd
from tqdm import tqdm
import os
from datasets import load_dataset
import re

class OllamaAfriXNLIEvaluator:
    def __init__(self, host='http://localhost:11434'):
        self.models = {
            "mistral:latest": "mistral:latest"
        }
        self.results = []
        self.host = host
    
    # Create prompts based on different strategies
    def create_prompts(self, premise, hypothesis, strategy_config):
        cot_enabled = strategy_config['cot_enabled']
        shot_type = strategy_config['shot_type']
        prompt_variant = strategy_config['prompt_variant']
        few_shot_examples = """
        Example 1:
        Premise: "The cat is sleeping on the mat."
        Hypothesis: "The cat is resting."
        Relationship: entailment

        Example 2:
        Premise: "It's raining outside."
        Hypothesis: "The sun is shining brightly."
        Relationship: contradiction
        
        Example 3:
        Premise: "She bought a red car."
        Hypothesis: "She purchased a vehicle."
        Relationship: neutral
        """

        base_prompts = {
            1: {
                'zero_shot': "Given the premise and hypothesis, determine their relationship. Choose from: entailment, neutral, contradiction.\n\nPremise: \"{premise}\"\nHypothesis: \"{hypothesis}\"\nRelationship:",
                'few_shot': "Determine the relationship between premise and hypothesis. Choose from: entailment, neutral, contradiction.\n{few_shot_examples}\nNow classify this:\nPremise: \"{premise}\"\nHypothesis: \"{hypothesis}\"\nRelationship:"
            },
            2: {
                'zero_shot': "Analyze the logical relationship between these two statements:\nPremise: \"{premise}\"\nHypothesis: \"{hypothesis}\"\n\nDoes the premise entail the hypothesis, are they neutral, or contradictory? Answer with one word: entailment, neutral, or contradiction.",
                'few_shot': "Classify the relationship type between statements:\n{few_shot_examples}\n\nYour turn:\nPremise: \"{premise}\"\nHypothesis: \"{hypothesis}\"\nRelationship type:"
            },
            3: {
                'zero_shot': "For the given pair of sentences, identify if the relationship is:\n- entailment (if premise supports hypothesis)\n- neutral (if premise doesn't strongly support or contradict)\n- contradiction (if premise contradicts hypothesis)\n\nPremise: {premise}\nHypothesis: {hypothesis}\nYour answer:",
                'few_shot': "Learn from examples and classify:\n{few_shot_examples}\n\nNew case:\nPremise: {premise}\nHypothesis: {hypothesis}\nClassification:"
            }
        }
        cot_instruction = "\n\nPlease reason step by step before giving your final answer."
        prompt_template = base_prompts[prompt_variant][shot_type]
        
        if shot_type == 'few_shot':
            prompt = prompt_template.format(
                few_shot_examples=few_shot_examples,
                premise=premise,
                hypothesis=hypothesis
            )
        else:
            prompt = prompt_template.format(
                premise=premise,
                hypothesis=hypothesis
            )
        if cot_enabled:
            prompt += cot_instruction
        return prompt
    
    
    # Extract predicted label from model response
    def extract_prediction_from_response(self, response):
        if not response:
            return None
        response_lower = response.lower().strip()
        if 'entailment' in response_lower:
            return 'entailment'
        elif 'contradiction' in response_lower:
            return 'contradiction'
        elif 'neutral' in response_lower:
            return 'neutral'
        patterns = [
            r'\b(entailment|contradiction|neutral)\b',
            r'answer:\s*(\w+)',
            r'relationship:\s*(\w+)',
            r'classification:\s*(\w+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, response_lower)
            if match:
                predicted = match.group(1) if match.lastindex else match.group(0)
                if predicted in ['entailment', 'contradiction', 'neutral']:
                    return predicted
        
        # If no clear match found, return the first 3 words as potential answer
        words = response_lower.split()[:3]
        for word in words:
            if word in ['entailment', 'contradiction', 'neutral']:
                return word
        return None
    

    # Query Ollama model with retry logic
    def query_ollama(self, model_name, prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = ollama.generate(model=model_name, prompt=prompt)
                end_time = time.time()
                
                if response and 'response' in response:
                    return {
                        "response": response['response'].strip(),
                        "processing_time": end_time - start_time,
                        "success": True
                    }
                else:
                    return {
                        "response": "Error: Empty response from model",
                        "processing_time": end_time - start_time,
                        "success": False
                    }
            except Exception as e:
                print(f"  Attempt {attempt + 1} failed for {model_name}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                continue
        return {
            "response": f"Error: All attempts failed",
            "processing_time": 0,
            "success": False
        }
    

    # Evaluate all prompt strategies on AfriXNLI dataset
    def evaluate_strategies(self, num_samples=1, language='swa'):
        print(f"Loading AfriXNLI dataset for language: {language}")
        dataset = load_dataset("masakhane/afrixnli", language)
        test_data = dataset["test"]
        num_samples = min(num_samples, len(test_data))
        print(f"Testing on {num_samples} samples...")
        strategies = []
        for cot in [True, False]:
            for shot_type in ['zero_shot', 'few_shot']:
                for variant in [1, 2, 3]:
                    strategies.append({
                        'cot_enabled': cot,
                        'shot_type': shot_type,
                        'prompt_variant': variant,
                        'strategy_name': f"{'CoT_' if cot else 'Direct_'}{shot_type}_v{variant}"
                    })
        print(f"Testing {len(strategies)} prompt strategies...")
        
        for sample_idx in tqdm(range(num_samples)):
            sample = test_data[sample_idx]
            premise = sample['premise']
            hypothesis = sample['hypothesis']
            true_label = sample['label']
            label_map = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
            true_label_text = label_map.get(true_label, 'unknown')
            for strategy in strategies:
                prompt = self.create_prompts(premise, hypothesis, strategy)
                for model_name in self.models.keys():
                    result = self.query_ollama(model_name, prompt)
                    predicted_label = None
                    is_correct = False
                    if result['success']:
                        predicted_label = self.extract_prediction_from_response(result['response'])
                        is_correct = (predicted_label == true_label_text)
                    self.results.append({
                        'sample_id': sample_idx,
                        'model': model_name,
                        'strategy_name': strategy['strategy_name'],
                        'cot_enabled': strategy['cot_enabled'],
                        'shot_type': strategy['shot_type'],
                        'prompt_variant': strategy['prompt_variant'],
                        'premise': premise,
                        'hypothesis': hypothesis,
                        'true_label': true_label_text,
                        'predicted_label': predicted_label,
                        'is_correct': is_correct,
                        'processing_time': result['processing_time'],
                        'success': result['success'],
                        'response': result['response'][:500] if result['success'] else result['response']  # Truncate long responses
                    })
                    time.sleep(1)
        return self.results
    

    # Generates comprehensive report of results
    def generate_comprehensive_report(self):
        if not self.results:
            print("No results to analyze.")
            return
        df = pd.DataFrame(self.results)
        successful_df = df[df['success']]
        print("\n" + "="*80)
        print(f"Total samples: {len(df['sample_id'].unique())}")
        print(f"Total evaluations: {len(df)}")
        print(f"Successful evaluations: {len(successful_df)}")
        if len(successful_df) > 0:
            overall_accuracy = successful_df['is_correct'].mean()
            print(f"Overall accuracy: {overall_accuracy:.2%}")
        print(f"\nStrategy-wise Performance:")
        print(f"{'STRATEGY':<25} {'ACCURACY':<10} {'SAMPLES':<10}")
        print("-" * 50)
        strategies = successful_df['strategy_name'].unique()
        for strategy in sorted(strategies):
            strategy_data = successful_df[successful_df['strategy_name'] == strategy]
            accuracy = strategy_data['is_correct'].mean()
            count = len(strategy_data)
            print(f"{strategy:<25} {accuracy:.2%}     {count:<10}")
        
        print(f"\nChain-of-Thought vs Direct:")
        cot_data = successful_df[successful_df['cot_enabled'] == True]
        direct_data = successful_df[successful_df['cot_enabled'] == False]
        if len(cot_data) > 0 and len(direct_data) > 0:
            cot_accuracy = cot_data['is_correct'].mean()
            direct_accuracy = direct_data['is_correct'].mean()
            print(f"CoT Accuracy:    {cot_accuracy:.2%} ({len(cot_data)} samples)")
            print(f"Direct Accuracy: {direct_accuracy:.2%} ({len(direct_data)} samples)")
            print(f"Difference:      {cot_accuracy - direct_accuracy:+.3f}")
        
        print(f"\nZero-shot vs Few-shot:")
        zero_shot_data = successful_df[successful_df['shot_type'] == 'zero_shot']
        few_shot_data = successful_df[successful_df['shot_type'] == 'few_shot']
        if len(zero_shot_data) > 0 and len(few_shot_data) > 0:
            zero_shot_accuracy = zero_shot_data['is_correct'].mean()
            few_shot_accuracy = few_shot_data['is_correct'].mean()
            print(f"Zero-shot Accuracy: {zero_shot_accuracy:.2%} ({len(zero_shot_data)} samples)")
            print(f"Few-shot Accuracy:  {few_shot_accuracy:.2%} ({len(few_shot_data)} samples)")
            print(f"Difference:         {few_shot_accuracy - zero_shot_accuracy:+.3f}")
        
        print(f"\nPrompt Variant Performance:")
        for variant in [1, 2, 3]:
            variant_data = successful_df[successful_df['prompt_variant'] == variant]
            if len(variant_data) > 0:
                accuracy = variant_data['is_correct'].mean()
                print(f"Variant {variant}: {accuracy:.2%} ({len(variant_data)} samples)")
        
        print(f"\nModel-wise Performance:")
        for model in successful_df['model'].unique():
            model_data = successful_df[successful_df['model'] == model]
            accuracy = model_data['is_correct'].mean()
            print(f"{model}: {accuracy:.2%} ({len(model_data)} samples)")
    

    # Save results to CSV file
    def save_results(self, filename="ollama_afrixnli_prompt_strategies.csv"):
        if not self.results:
            print("No results to save")
            return None
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        return df


def main():
    print("=== Prompt Strategies Evaluation on AfriXNLI Dataset using Ollama Models ===")
    evaluator = OllamaAfriXNLIEvaluator()
    results = evaluator.evaluate_strategies(num_samples=1, language='swa')
    evaluator.save_results("ollama_afrixnli_swa_results.csv")
    evaluator.generate_comprehensive_report()

if __name__ == "__main__":
    main()