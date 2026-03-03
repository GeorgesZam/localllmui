#!/usr/bin/env python3
"""
Improvement Engine for AI Responses.
Ce script détecte et améliore les réponses insatisfaisantes de l'IA.
"""

import re
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class ResponseImprover:
    """Classe pour améliorer les réponses de l'IA."""

    def __init__(self, rag_system=None):
        self.rag_system = rag_system
        self.response_threshold = 50  # Minimum mots pour une réponse acceptable

        # Mot-clés indiquant des réponses insatisfaisantes
        self.insufficient_indicators = [
            "i don't have this information",
            "i don't know",
            "i'm not sure",
            "i cannot find",
            "not available",
            "no information",
            "insufficient data",
            "i'm sorry",
            "unable to",
            "unfortunately",
        ]

        # Suggestions pour améliorer les questions
        self.question_improvements = {
            "what is": "Try 'How does [topic] work?' for more detailed information",
            "how do i": "Provide more context about your specific situation",
            "why": "Include examples or scenarios for better answers",
            "can i": "Specify what you're trying to achieve",
            "does it": "Ask about specific features or capabilities",
        }

    def is_insufficient_response(self, response: str, question: str = "") -> bool:
        """Vérifie si une réponse est insuffisante."""
        if not response or len(response.strip()) < 20:
            return True

        response_lower = response.lower()

        # Vérifier les indicateurs de réponse insuffisante
        for indicator in self.insufficient_indicators:
            if indicator in response_lower:
                logger.info(f"Réponse insuffisante détectée: '{indicator}'")
                return True

        # Vérifier la longueur
        word_count = len(response.split())
        if word_count < self.response_threshold:
            logger.warning(f"Réponse courte: {word_count} mots")
            return True

        # Vérifier si la réponse contient des placeholders vides
        empty_patterns = [
            r"\[\s*\]",  # [ ] ou []
            r"\{.*\}",   # {} contenant peu de contenu
        ]

        for pattern in empty_patterns:
            if re.search(pattern, response):
                return True

        return False

    def get_improvement_suggestions(self, question: str) -> Dict[str, str]:
        """Obtenir des suggestions pour améliorer la question."""
        suggestions = {}

        # Analyse de la question
        question_lower = question.lower()

        # Vérifier la longueur
        if len(question.split()) < 5:
            suggestions["length"] = "Your question is too short. Add more details."

        # Vérifier les questions générales
        if question_lower.startswith(("what is ", "what are ", "who ", "when ", "where ")):
            suggestions["specificity"] = "Try asking 'How' or 'Why' questions for more detailed answers."

        # Vérifier les mots de vocabulaire technique
        if any(word in question_lower for word in ["bug", "error", "problem", "issue"]):
            suggestions["technical"] = "Describe the specific error message or problem you're facing."

        # Vérifier si la demande est claire
        if not any(word in question_lower for word in ["how", "why", "what is", "when", "where", "who"]):
            suggestions["structure"] = "Make sure your question starts with a question word (How, What, Why, etc.)."

        return suggestions

    def improve_response(self, original_response: str, question: str,
                       available_context: Optional[Dict] = None) -> str:
        """Améliorer une réponse insuffisante."""
        if not self.is_insufficient_response(original_response, question):
            return original_response

        improvements = []

        # Amélioration basée sur le contexte RAG
        if self.rag_system and available_context:
            rag_improvement = self._improve_with_rag(question, available_context)
            if rag_improvement:
                improvements.append(rag_improvement)

        # Suggestions générales
        general_improvement = self._get_general_improvement(question)
        if general_improvement:
            improvements.append(general_improvement)

        # Amélioration basée sur le type de question
        question_improvement = self._improve_by_question_type(question)
        if question_improvement:
            improvements.append(question_improvement)

        # Combiner les améliorations
        if improvements:
            improved = "\n\n".join(improvements)
            return f"{original_response}\n\n Suggestions to get better answers:\n{improved}"

        return original_response

    def _improve_with_rag(self, question: str, context: Dict) -> Optional[str]:
        """Améliorer en utilisant le système RAG."""
        if not hasattr(self.rag_system, 'search'):
            return None

        try:
            # Chercher des documents pertinents
            search_results = self.rag_system.search(
                question,
                top_k=3,
                max_context_chars=500
            )

            if search_results and len(search_results[1]) > 0:
                return f"🔍 I found some relevant documents. You might want to check these first:\n" + \
                       " - " + "\n - ".join([doc.get('title', doc.get('filename', 'Document'))
                                            for doc in search_results[1][:2]])
        except Exception as e:
            logger.error(f"Erreur RAG: {e}")

        return None

    def _get_general_improvement(self, question: str) -> Optional[str]:
        """Amélioration générale pour les questions."""
        # Identifier les catégories communes
        if "python" in question.lower():
            return " For Python questions, include your version and error messages if applicable."

        if "config" in question.lower() or "setting" in question.lower():
            return " For configuration questions, mention your platform and what you're trying to achieve."

        if "install" in question.lower():
            return " For installation questions, include your operating system and any error messages."

        if not self.rag_system:
            return " Consider adding relevant documents to improve responses."

        return None

    def _improve_by_question_type(self, question: str) -> Optional[str]:
        """Amélioration basée sur le type de question."""
        question_lower = question.lower()

        # Questions techniques
        if any(word in question_lower for word in ["how", "why", "what", "which"]):
            return " Provide examples of what you've tried or what's not working."

        # Questions "Do I need..."
        if "do i need" in question_lower or "should i" in question_lower:
            return " Describe your use case or goal for better recommendations."

        # Questions "Is it possible..."
        if "is it possible" in question_lower:
            return " Describe your specific requirements or constraints."

        return None

    def generate_follow_up_questions(self, question: str, original_response: str) -> List[str]:
        """Générer des questions de suivi pertinentes."""
        follow_ups = []
        question_lower = question.lower()

        # Basé sur le type de question original
        if "how" in question_lower:
            follow_ups.append("Can you provide a step-by-step example?")
            follow_ups.append("What are the common mistakes to avoid?")

        if "what is" in question_lower:
            follow_ups.append("How can I implement this in practice?")
            follow_ups.append("What are the alternatives?")

        if "why" in question_lower:
            follow_ups.append("What happens if I don't do this?")
            follow_ups.append("Are there any side effects?")

        # Ajouter des questions basées sur la réponse insuffisante
        if "don't know" in original_response.lower() or "not sure" in original_response.lower():
            follow_ups.append("Where can I find more information about this?")
            follow_ups.append("Can you direct me to the official documentation?")

        return follow_ups[:3]  # Retourner jusqu'à 3 questions

def main():
    """Test du système d'amélioration de réponses."""
    print("=== Response Improvement Engine ===")

    # Initialiser
    improver = ResponseImprover()

    # Exemples de tests
    test_cases = [
        {
            "question": "How do I fix this bug?",
            "response": "I don't know what the bug is."
        },
        {
            "question": "What is the configuration?",
            "response": "I don't have this information."
        },
        {
            "question": "How to install Python?",
            "response": "You need to download it from python.org."
        }
    ]

    print("\n--- Test des réponses ---")
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Question: {case['question']}")
        print(f"Réponse originale: {case['response']}")

        # Vérifier si la réponse est insuffisante
        is_insufficient = improver.is_insufficient_response(case['response'], case['question'])
        print(f"Est insuffisante: {is_insufficient}")

        if is_insufficient:
            # Obtenir des suggestions
            suggestions = improver.get_improvement_suggestions(case['question'])
            print(f"Suggestions: {suggestions}")

            # Obtenir des questions de suivi
            follow_ups = improver.generate_follow_up_questions(case['question'], case['response'])
            print(f"Questions de suivi: {follow_ups}")

if __name__ == "__main__":
    main()