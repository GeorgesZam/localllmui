#!/usr/bin/env python3
"""
Response Handler for LocalLLMUI.
Ce module gère l'amélioration des réponses et la détection des questions non répondues.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from response_improver import ResponseImprover

logger = logging.getLogger(__name__)


class ResponseHandler:
    """Gestionnaire centralisé des réponses."""

    def __init__(self, rag_system=None):
        self.improver = ResponseImprover(rag_system)
        self.response_history = []
        self.unanswered_count = 0

    def process_response(
        self, question: str, response: str, context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Traite une réponse et retourne le résultat amélioré si nécessaire.

        Args:
            question: La question posée
            response: La réponse brute de l'IA
            context: Contexte supplémentaire (conversation, RAG, etc.)

        Returns:
            Dictionnaire contenant la réponse améliorée et métadonnées
        """
        # Stocker dans l'historique
        entry = {
            "question": question,
            "response": response,
            "timestamp": self._get_timestamp(),
            "context": context or {},
        }
        self.response_history.append(entry)

        # Vérifier si la réponse est insuffisante
        is_insufficient = self.improver.is_insufficient_response(response, question)

        if is_insufficient:
            self.unanswered_count += 1
            logger.info(
                f"Question non répondue #{self.unanswered_count}: {question[:50]}..."
            )

            # Améliorer la réponse
            improved_response = self.improver.improve_response(
                response, question, context
            )

            # Obtenir des suggestions
            suggestions = self.improver.get_improvement_suggestions(question)

            # Obtenir des questions de suivi
            follow_ups = self.improver.generate_follow_up_questions(question, response)

            return {
                "response": improved_response,
                "original_response": response,
                "improved": True,
                "is_insufficient": True,
                "suggestions": suggestions,
                "follow_ups": follow_ups,
                "unanswered_count": self.unanswered_count,
            }
        else:
            # Réponse satisfaisante, mais peut encore suggérer des questions de suivi
            follow_ups = self.improver.generate_follow_up_questions(question, response)

            return {
                "response": response,
                "original_response": response,
                "improved": False,
                "is_insufficient": False,
                "suggestions": {},
                "follow_ups": follow_ups,
                "unanswered_count": self.unanswered_count,
            }

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Obtenir des métriques sur la qualité des réponses."""
        total_responses = len(self.response_history)
        if total_responses == 0:
            return {"total": 0, "unanswered_rate": 0}

        unanswered_rate = (self.unanswered_count / total_responses) * 100

        # Analyser les motifs communs
        common_patterns = self._analyze_common_patterns()

        return {
            "total": total_responses,
            "unanswered": self.unanswered_count,
            "unanswered_rate": unanswered_rate,
            "success_rate": 100 - unanswered_rate,
            "common_patterns": common_patterns,
            "average_response_length": self._calculate_average_length(),
        }

    def _get_timestamp(self) -> str:
        """Obtenir un timestamp formatté."""
        from datetime import datetime

        return datetime.now().isoformat()

    def _analyze_common_patterns(self) -> List[Dict[str, Any]]:
        """Analyser les motifs communs dans les questions non répondues."""
        if self.unanswered_count == 0:
            return []

        # Extraire les questions non répondues
        unanswered_questions = [
            entry["question"]
            for entry in self.response_history
            if self.improver.is_insufficient_response(
                entry["response"], entry["question"]
            )
        ]

        # Compter les mots
        word_counts = {}
        for question in unanswered_questions:
            words = re.findall(r"\b\w+\b", question.lower())
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Retourner les mots les plus fréquents
        return [
            {"pattern": word, "count": count}
            for word, count in sorted(
                word_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]
        ]

    def _calculate_average_length(self) -> float:
        """Calculer la longueur moyenne des réponses."""
        if not self.response_history:
            return 0

        total_length = sum(
            len(entry["response"].split()) for entry in self.response_history
        )
        return total_length / len(self.response_history)

    def suggest_improvements_for_ui(self, question: str) -> Dict[str, str]:
        """Suggérer des améliorations spécifiques pour l'interface utilisateur."""
        return self.improver.get_improvement_suggestions(question)

    def should_show_rag_tip(self, question: str) -> bool:
        """Déterminer si on devrait afficher un tip RAG."""
        question_lower = question.lower()

        # Motifs indiquant que RAG pourrait aider
        rag_indicators = [
            "how to",
            "what is",
            "explain",
            "tutorial",
            "guide",
            "best practice",
            "example",
            "documentation",
        ]

        # Vérifier si la question est de type recherche d'information
        is_information_query = any(
            indicator in question_lower for indicator in rag_indicators
        )

        # Vérifier si la réponse précédente était insuffisante
        recent_insufficient = len(
            self.response_history
        ) > 0 and self.improver.is_insufficient_response(
            self.response_history[-1]["response"], self.response_history[-1]["question"]
        )

        return is_information_query or recent_insufficient


class ConversationAwareResponseHandler(ResponseHandler):
    """Gestionnaire de réponses conscient du contexte de la conversation."""

    def __init__(self, rag_system=None):
        super().__init__(rag_system)
        self.conversation_history = {}  # {conversation_id: [messages]}
        self.current_conversation_id = None

    def set_conversation(self, conversation_id: str):
        """Définir la conversation actuelle."""
        self.current_conversation_id = conversation_id
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []

    def add_to_conversation(self, question: str, response: str):
        """Ajouter un échange à l'historique de la conversation."""
        if self.current_conversation_id:
            self.conversation_history[self.current_conversation_id].append(
                {
                    "question": question,
                    "response": response,
                    "timestamp": self._get_timestamp(),
                }
            )

    def get_conversation_context(self, conversation_id: str = None) -> Dict[str, Any]:
        """Obtenir le contexte d'une conversation."""
        cid = conversation_id or self.current_conversation_id
        if not cid:
            return {}

        history = self.conversation_history.get(cid, [])

        # Extraire des thèmes communs
        topics = self._extract_topics(history)

        # Identifier les problèmes récurrents
        recurring_issues = self._identify_recurring_issues(history)

        return {
            "length": len(history),
            "topics": topics,
            "recurring_issues": recurring_issues,
            "last_exchange": history[-1] if history else None,
        }

    def _extract_topics(self, history: List[Dict]) -> List[str]:
        """Extraire les thèmes d'une conversation."""
        topics = []
        common_topic_words = ["python", "config", "error", "install", "bug", "feature"]

        for entry in history:
            question_lower = entry["question"].lower()
            for topic in common_topic_words:
                if topic in question_lower:
                    topics.append(topic)
                    break

        return list(set(topics))  # Retirer les doublons

    def _identify_recurring_issues(self, history: List[Dict]) -> List[str]:
        """Identifier les problèmes récurrents."""
        issues = []

        # Motifs de problèmes courants
        problem_indicators = [
            ("error", "Erreurs techniques"),
            ("bug", "Bugs"),
            ("not working", "Problèmes de fonctionnement"),
            ("doesn't work", "Problèmes de fonctionnement"),
            ("failed", "Échecs"),
            ("timeout", "Problèmes de timeout"),
        ]

        for entry in history:
            question_lower = entry["question"].lower()
            for indicator, issue_name in problem_indicators:
                if indicator in question_lower:
                    issues.append(issue_name)
                    break

        return list(set(issues))  # Retirer les doublons


# Exemple d'utilisation
def main():
    """Test du gestionnaire de réponses."""
    print("=== Response Handler Test ===")

    handler = ResponseHandler()

    # Simuler quelques échanges
    test_cases = [
        {
            "question": "How do I install Python packages?",
            "response": "I don't have this information.",
        },
        {
            "question": "What is the configuration file?",
            "response": "The configuration file is config.py",
        },
        {
            "question": "How to fix a bug?",
            "response": "I don't know what bug you're referring to.",
        },
    ]

    print("\n--- Processing test cases ---")
    for i, case in enumerate(test_cases, 1):
        result = handler.process_response(case["question"], case["response"])

        print(f"\nTest {i}:")
        print(f"Question: {case['question']}")
        print(f"Original: {case['response']}")
        print(f"Improved: {result['response']}")
        print(f"Is insufficient: {result['is_insufficient']}")

        if result["suggestions"]:
            print(f"Suggestions: {result['suggestions']}")


if __name__ == "__main__":
    main()
