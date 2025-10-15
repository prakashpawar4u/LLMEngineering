# llm_evaluator.py
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class LLMEvaluator:
    """Comprehensive LLM performance evaluation framework"""
    
    def __init__(self):
        self.metrics_history = []
    
    def evaluate_rag_system(self, 
                          questions: List[str],
                          ground_truths: List[str],
                          llm_answers: List[str],
                          contexts: List[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate RAG system performance comprehensively
        
        Args:
            questions: List of test questions
            ground_truths: List of ground truth answers
            llm_answers: List of LLM generated answers
            contexts: List of context chunks used for each answer
        """
        results = {}
        
        # Basic accuracy metrics
        results.update(self._calculate_accuracy_metrics(ground_truths, llm_answers))
        
        # Semantic similarity metrics
        results.update(self._calculate_semantic_similarity(ground_truths, llm_answers))
        
        # Hallucination detection
        if contexts:
            results.update(self._detect_hallucinations(llm_answers, contexts))
        
        # Faithfulness metrics
        if contexts:
            results.update(self._calculate_faithfulness(llm_answers, contexts))
        
        # Save results
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'results': results,
            'sample_size': len(questions)
        })
        
        return results
    
    def _calculate_accuracy_metrics(self, ground_truths: List[str], llm_answers: List[str]) -> Dict[str, float]:
        """Calculate traditional accuracy metrics"""
        # For exact match (strict evaluation)
        exact_matches = [1 if gt.lower() == ans.lower() else 0 
                        for gt, ans in zip(ground_truths, llm_answers)]
        exact_match_accuracy = np.mean(exact_matches)
        
        # For partial match (lenient evaluation)
        partial_matches = [1 if gt.lower() in ans.lower() or ans.lower() in gt.lower() else 0 
                          for gt, ans in zip(ground_truths, llm_answers)]
        partial_match_accuracy = np.mean(partial_matches)
        
        return {
            'exact_match_accuracy': exact_match_accuracy,
            'partial_match_accuracy': partial_match_accuracy,
            'total_questions': len(ground_truths)
        }
    
    def _calculate_semantic_similarity(self, ground_truths: List[str], llm_answers: List[str]) -> Dict[str, float]:
        """Calculate semantic similarity using embeddings"""
        try:
            from sentence_transformers import SentenceTransformer, util
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            gt_embeddings = model.encode(ground_truths, convert_to_tensor=True)
            ans_embeddings = model.encode(llm_answers, convert_to_tensor=True)
            
            # Calculate cosine similarities
            similarities = util.cos_sim(gt_embeddings, ans_embeddings).diagonal().cpu().numpy()
            
            return {
                'semantic_similarity_mean': float(np.mean(similarities)),
                'semantic_similarity_std': float(np.std(similarities)),
                'semantic_similarity_min': float(np.min(similarities)),
                'semantic_similarity_max': float(np.max(similarities))
            }
        except ImportError:
            logger.warning("Sentence-transformers not available, skipping semantic similarity")
            return {}
    
    def _detect_hallucinations(self, answers: List[str], contexts: List[List[str]]) -> Dict[str, float]:
        """Detect and quantify hallucinations"""
        hallucination_scores = []
        
        for answer, context_chunks in zip(answers, contexts):
            # Combine all context for this question
            full_context = " ".join(context_chunks)
            score = self._calculate_hallucination_score(answer, full_context)
            hallucination_scores.append(score)
        
        return {
            'hallucination_rate': np.mean([1 if score > 0.7 else 0 for score in hallucination_scores]),
            'avg_hallucination_score': float(np.mean(hallucination_scores)),
            'hallucination_std': float(np.std(hallucination_scores))
        }
    
    def _calculate_hallucination_score(self, answer: str, context: str) -> float:
        """Calculate hallucination score for a single answer"""
        # Simple keyword-based approach (can be enhanced with ML models)
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        # Words in answer but not in context (potential hallucinations)
        novel_words = answer_words - context_words
        
        # Remove common words that might not indicate hallucination
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        novel_words = novel_words - common_words
        
        if not answer_words:
            return 0.0
        
        hallucination_ratio = len(novel_words) / len(answer_words)
        return hallucination_ratio
    
    def _calculate_faithfulness(self, answers: List[str], contexts: List[List[str]]) -> Dict[str, float]:
        """Calculate how faithful answers are to the provided context"""
        faithfulness_scores = []
        
        for answer, context_chunks in zip(answers, contexts):
            full_context = " ".join(context_chunks)
            score = self._calculate_faithfulness_score(answer, full_context)
            faithfulness_scores.append(score)
        
        return {
            'faithfulness_mean': float(np.mean(faithfulness_scores)),
            'faithfulness_std': float(np.std(faithfulness_scores))
        }
    
    def _calculate_faithfulness_score(self, answer: str, context: str) -> float:
        """Calculate faithfulness score for a single answer"""
        # Simple overlap-based approach
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        common_words = answer_words & context_words
        
        if not answer_words:
            return 0.0
        
        faithfulness_ratio = len(common_words) / len(answer_words)
        return faithfulness_ratio
    
    def generate_evaluation_report(self, results: Dict[str, float]) -> str:
        """Generate a comprehensive evaluation report"""
        report = [
            "LLM RAG System Evaluation Report",
            "=" * 40,
            f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Sample Size: {results.get('total_questions', 'N/A')}",
            "",
            "Performance Metrics:",
            f"  • Exact Match Accuracy: {results.get('exact_match_accuracy', 0):.3f}",
            f"  • Partial Match Accuracy: {results.get('partial_match_accuracy', 0):.3f}",
            f"  • Semantic Similarity: {results.get('semantic_similarity_mean', 0):.3f}",
            "",
            "Hallucination Analysis:",
            f"  • Hallucination Rate: {results.get('hallucination_rate', 0):.3f}",
            f"  • Avg Hallucination Score: {results.get('avg_hallucination_score', 0):.3f}",
            "",
            "Faithfulness Metrics:",
            f"  • Faithfulness Score: {results.get('faithfulness_mean', 0):.3f}",
        ]
        
        return "\n".join(report)

# Example usage
def create_test_dataset() -> Tuple[List[str], List[str]]:
    """Create a simple test dataset"""
    questions = [
        "What is the company's refund policy?",
        "How long does shipping take?",
        "What payment methods do you accept?",
        "What is the warranty period for products?",
    ]
    
    ground_truths = [
        "The company offers a 30-day refund policy for all products.",
        "Standard shipping takes 5-7 business days, express shipping takes 2-3 days.",
        "We accept credit cards, PayPal, and bank transfers.",
        "All products come with a 1-year manufacturer warranty.",
    ]
    
    return questions, ground_truths