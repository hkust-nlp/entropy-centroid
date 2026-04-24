"""
Dataset loaders for Logic reasoning tasks.

Supports:
- KOR-Bench: Local JSONL data from KOR-Bench directory
- SynLogic: HuggingFace dataset (MiniMaxAI/SynLogic)
"""

import json
import os
import sys
from typing import Dict, List, Optional, Union
from datasets import load_dataset
from tqdm import tqdm

# Add KOR-Bench to path for using its utilities
KORBENCH_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'KOR-Bench')
if os.path.exists(KORBENCH_PATH):
    sys.path.insert(0, KORBENCH_PATH)


class KorBenchDatasetLoader:
    """
    Loader for KOR-Bench dataset from local JSONL files.
    
    Supports categories: cipher, logic, operation, puzzle, counterfactual
    Supports loading multiple categories at once.
    """
    
    VALID_CATEGORIES = ['cipher', 'logic', 'operation', 'puzzle', 'counterfactual']
    
    def __init__(
        self,
        korbench_path: str,
        category: Union[str, List[str]] = 'all',
        mode: str = 'zero-shot',
        max_samples: Optional[int] = None,
    ):
        """
        Initialize KOR-Bench dataset loader.
        
        Args:
            korbench_path: Path to KOR-Bench directory
            category: Category to load. Can be:
                - "all": Load all categories (default)
                - Single string: "cipher", "logic", "operation", "puzzle", "counterfactual"
                - List of strings: ["cipher", "logic"] to load multiple categories
            mode: Prompt mode (zero-shot, three-shot, trick, self-correction)
            max_samples: Maximum number of samples to load per category (None for all)
        """
        self.korbench_path = korbench_path
        self.mode = mode
        self.max_samples = max_samples
        self.config_path = os.path.join(korbench_path, 'config', 'prompt')
        
        # Parse categories
        if category == 'all' or category is None:
            self.categories = self.VALID_CATEGORIES.copy()
        elif isinstance(category, str):
            if category not in self.VALID_CATEGORIES:
                raise ValueError(f"Invalid category: {category}. Must be one of {self.VALID_CATEGORIES} or 'all'")
            self.categories = [category]
        elif isinstance(category, list):
            for cat in category:
                if cat not in self.VALID_CATEGORIES:
                    raise ValueError(f"Invalid category: {cat}. Must be one of {self.VALID_CATEGORIES}")
            self.categories = category
        else:
            raise ValueError(f"category must be 'all', a string, or a list of strings")
        
    def _read_jsonl(self, filepath: str) -> List[Dict]:
        """Read JSONL file and return list of records."""
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def _read_yaml(self, config_name: str) -> Dict:
        """Read YAML config file."""
        import yaml
        yaml_path = os.path.join(self.config_path, f'{config_name}.yaml')
        with open(yaml_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _read_rules(self, category: str) -> Dict[str, Dict]:
        """Read rules indexed by rule_id for a specific category."""
        data_path = os.path.join(self.korbench_path, 'data', category)
        rule_file = os.path.join(data_path, 'rule.jsonl')
        rules_list = self._read_jsonl(rule_file)
        return {item['idx']: item for item in rules_list}
    
    def _format_prompt(self, rule_content: str, question: str, template: Dict, category: str) -> str:
        """Format prompt using template."""
        prompt_key = f'{category}_prompt_format'
        if prompt_key in template and template[prompt_key]:
            return template[prompt_key][0].format(rule_content, question)
        return f"Rule:\n{rule_content}\n\nQuestion:\n{question}"
    
    def _load_category(self, category: str, template: Dict) -> List[Dict]:
        """Load samples for a single category."""
        data_path = os.path.join(self.korbench_path, 'data', category)
        
        rules = self._read_rules(category)
        sample_file = os.path.join(data_path, 'sample.jsonl')
        samples = self._read_jsonl(sample_file)
        
        if self.max_samples is not None:
            samples = samples[:self.max_samples]
        
        formatted_samples = []
        for idx, sample in enumerate(samples):
            rule_id = sample.get('rule_id')
            rule = rules.get(rule_id, {})
            rule_content = rule.get('rule_content', '')
            question = sample.get('question', '')
            
            prompt = self._format_prompt(rule_content, question, template, category)
            
            formatted_sample = {
                'id': f"korbench_{category}_{sample.get('idx', idx)}",
                'problem': question,
                'solution': sample.get('answer', ''),
                'prompt': prompt,
                'rule_id': rule_id,
                'rule_content': rule_content,
                'category': category,
                'source': 'korbench',
                'needle': sample.get('needle', []),
                'original_data': sample,
            }
            formatted_samples.append(formatted_sample)
        
        return formatted_samples
    
    def load(self) -> List[Dict]:
        """Load the KOR-Bench dataset for all specified categories."""
        categories_str = ', '.join(self.categories)
        print(f"Loading KOR-Bench dataset: [{categories_str}] ({self.mode})")
        
        config_name = self.mode
        if self.mode in ['self-correction', 'self-correction-with-needle']:
            config_name = 'zero-shot'
        template = self._read_yaml(config_name)
        
        all_samples = []
        for category in self.categories:
            print(f"  Loading category: {category}")
            category_samples = self._load_category(category, template)
            print(f"    Loaded {len(category_samples)} samples from {category}")
            all_samples.extend(category_samples)
        
        print(f"Total loaded: {len(all_samples)} samples from {len(self.categories)} categories")
        return all_samples


class SynLogicDatasetLoader:
    """
    Loader for SynLogic dataset from HuggingFace.
    
    Dataset: MiniMaxAI/SynLogic
    Available subsets: 'easy', 'hard'
    
    Note: The dataset format changed in late 2025. New format uses:
    - data_source: Contains task path (e.g., 'val/campsite')
    - prompt: Contains the problem in conversation format
    - extra_info: Contains game_data_str, index, metadata, etc.
    """
    
    VALID_SUBSETS = ['easy', 'hard']
    
    def __init__(
        self,
        task_name: Optional[str] = None,
        split: str = 'validation',
        subset: str = 'hard',
        max_samples: Optional[int] = None,
        synlogic_path: Optional[str] = None,
    ):
        """
        Initialize SynLogic dataset loader.
        
        Args:
            task_name: Specific task to filter (e.g., 'campsite', 'sudoku', None for all)
            split: Dataset split to load (train, validation, test)
            subset: Dataset subset/config to load ('easy' or 'hard', default: 'hard')
            max_samples: Maximum number of samples to load (None for all)
            synlogic_path: Path to SynLogic directory (for verifier import)
        """
        self.task_name = task_name
        self.split = split
        self.subset = subset if subset in self.VALID_SUBSETS else 'hard'
        self.max_samples = max_samples
        self.synlogic_path = synlogic_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
            'SynLogic'
        )
        
        if os.path.exists(self.synlogic_path):
            sys.path.insert(0, self.synlogic_path)
    
    def _parse_game_data(self, game_data_str: str) -> Dict:
        """Parse game_data_str JSON string."""
        if not game_data_str:
            return {}
        try:
            return json.loads(game_data_str)
        except (json.JSONDecodeError, TypeError):
            return {}
    
    def _extract_task_name(self, data_source: str, source_file: str = None) -> Optional[str]:
        """
        Extract task name from data_source or source_file field.
        
        Args:
            data_source: New format field (e.g., 'val/campsite')
            source_file: Old format field (e.g., 'campsite/xxx.json')
            
        Returns:
            Task name in lowercase
        """
        # Try new format first: data_source like 'val/campsite'
        if data_source:
            parts = data_source.split('/')
            if len(parts) >= 2:
                return parts[-1].lower()  # Get the task name part
            elif len(parts) == 1:
                return parts[0].lower()
        
        # Fallback to old format: source_file like 'campsite/xxx.json'
        if source_file:
            parts = source_file.split('/')
            if parts:
                return parts[0].lower()
        
        return None
    
    def _extract_prompt_text(self, prompt_data) -> str:
        """
        Extract text from prompt field (handles both string and conversation format).
        
        Args:
            prompt_data: Either a string or a list of conversation turns
            
        Returns:
            Extracted prompt text
        """
        if isinstance(prompt_data, str):
            return prompt_data
        
        if isinstance(prompt_data, list):
            # Conversation format: [{'content': '...', 'role': 'user'}, ...]
            for turn in prompt_data:
                if isinstance(turn, dict) and turn.get('role') == 'user':
                    return turn.get('content', '')
            # Fallback: join all content
            return '\n'.join(
                turn.get('content', str(turn)) 
                for turn in prompt_data 
                if isinstance(turn, dict)
            )
        
        return str(prompt_data) if prompt_data else ''
    
    def load(self) -> List[Dict]:
        """Load the SynLogic dataset from HuggingFace."""
        print(f"Loading SynLogic dataset from HuggingFace (subset: {self.subset}, split: {self.split})")
        if self.task_name:
            print(f"Filtering for task: {self.task_name}")
        
        try:
            # Load dataset with required subset/config name
            dataset = load_dataset("MiniMaxAI/SynLogic", self.subset, split=self.split)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise ValueError(f"Failed to load SynLogic dataset with subset='{self.subset}', split='{self.split}'. "
                           f"Available subsets: {self.VALID_SUBSETS}")
        
        print(f"Loaded {len(dataset)} samples from HuggingFace")
        print(f"Dataset columns: {dataset.column_names}")
        
        # Detect dataset format (new vs old)
        sample_columns = dataset.column_names
        is_new_format = 'extra_info' in sample_columns or 'data_source' in sample_columns
        
        if is_new_format:
            print("Detected new dataset format (with extra_info/data_source)")
        else:
            print("Detected old dataset format (with game_data_str/source_file)")
        
        # Filter by task name if specified
        if self.task_name:
            filtered_indices = []
            for i, sample in enumerate(dataset):
                if is_new_format:
                    # New format: check data_source
                    data_source = sample.get('data_source', '')
                    extra_info = sample.get('extra_info', {})
                    if isinstance(extra_info, dict):
                        source_file = extra_info.get('source_file', '')
                    else:
                        source_file = ''
                else:
                    # Old format
                    data_source = ''
                    source_file = sample.get('source_file', '')
                
                task = self._extract_task_name(data_source, source_file)
                if task and self.task_name.lower() in task.lower():
                    filtered_indices.append(i)
            
            dataset = dataset.select(filtered_indices)
            print(f"Filtered to {len(dataset)} samples for task: {self.task_name}")
        
        if self.max_samples is not None:
            dataset = dataset.select(range(min(self.max_samples, len(dataset))))
            print(f"Limited to {len(dataset)} samples")
        
        formatted_samples = []
        task_counts = {}
        
        for idx, sample in enumerate(tqdm(dataset, desc="Formatting SynLogic samples")):
            if is_new_format:
                # New format: extract from extra_info
                extra_info = sample.get('extra_info', {})
                if isinstance(extra_info, str):
                    try:
                        extra_info = json.loads(extra_info)
                    except:
                        extra_info = {}
                
                game_data_str = extra_info.get('game_data_str', '{}')
                game_data = self._parse_game_data(game_data_str)
                
                # Get question and answer
                question = game_data.get('question', '') or extra_info.get('original_question', '')
                metadata = game_data.get('metadata', {}) or extra_info.get('metadata', {})
                difficulty = game_data.get('difficulty', 1)
                
                # Answer can be in multiple places - check all possibilities
                answer = game_data.get('answer', '')
                if not answer:
                    answer = extra_info.get('original_answer', '')
                if not answer and metadata:
                    # Some tasks store solution in metadata
                    solution_in_meta = metadata.get('solution', '')
                    if solution_in_meta:
                        # Convert to string if it's a list/dict
                        if isinstance(solution_in_meta, (list, dict)):
                            answer = json.dumps(solution_in_meta)
                        else:
                            answer = str(solution_in_meta)
                
                # Extract task name
                data_source = sample.get('data_source', '')
                source_file = extra_info.get('source_file', '')
                task_name = self._extract_task_name(data_source, source_file) or 'unknown'
                
                # Get sample index
                sample_id = extra_info.get('index', str(idx))
                
                # Extract prompt text from conversation format
                prompt_data = sample.get('prompt', '')
                prompt = self._extract_prompt_text(prompt_data)
                
                # Use question as prompt if prompt is empty
                if not prompt and question:
                    prompt = question
                
            else:
                # Old format (backward compatibility)
                game_data_str = sample.get('game_data_str', '{}')
                game_data = self._parse_game_data(game_data_str)
                
                question = game_data.get('question', '')
                answer = game_data.get('answer', '')
                metadata = game_data.get('metadata', {})
                difficulty = game_data.get('difficulty', 1)
                
                source_file = sample.get('source_file', '')
                task_name = self._extract_task_name('', source_file) or 'unknown'
                sample_id = sample.get('index', str(idx))
                
                prompt = question
            
            # Track task counts
            task_counts[task_name] = task_counts.get(task_name, 0) + 1
            
            formatted_sample = {
                'id': f"synlogic_{task_name}_{sample_id}",
                'problem': question,
                'solution': answer,
                'prompt': prompt,
                'task_name': task_name,
                'difficulty': difficulty,
                'metadata': metadata,
                'source': 'synlogic',
                'game_data': game_data,
                'original_data': dict(sample),
            }
            formatted_samples.append(formatted_sample)
        
        # Print task distribution
        print(f"\nTask distribution ({len(task_counts)} tasks):")
        for task, count in sorted(task_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {task}: {count}")
        if len(task_counts) > 10:
            print(f"  ... and {len(task_counts) - 10} more tasks")
        
        return formatted_samples


def create_logic_dataset_loader(config: Dict):
    """
    Create a logic dataset loader based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dataset loader instance
    """
    dataset_config = config.get('dataset', {})
    dataset_type = dataset_config.get('type', 'synlogic')
    
    if dataset_type == 'korbench':
        korbench_config = dataset_config.get('korbench', {})
        return KorBenchDatasetLoader(
            korbench_path=korbench_config.get('path', './KOR-Bench'),
            category=korbench_config.get('category', 'logic'),
            mode=korbench_config.get('mode', 'zero-shot'),
            max_samples=dataset_config.get('max_samples'),
        )
    elif dataset_type == 'synlogic':
        synlogic_config = dataset_config.get('synlogic', {})
        return SynLogicDatasetLoader(
            task_name=synlogic_config.get('task_name'),
            split=dataset_config.get('split', 'validation'),
            subset=synlogic_config.get('subset', 'hard'),
            max_samples=dataset_config.get('max_samples'),
            synlogic_path=synlogic_config.get('path', './SynLogic'),
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
