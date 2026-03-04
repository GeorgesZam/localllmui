#!/usr/bin/env python3
"""
Script pour analyser et gérer les questions non répondues dans l'application.
Ce script détecte quand l'IA ne répond pas correctement à une question
et propose des alternatives ou des réponses de secours.
"""

import re
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unanswered_questions.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class UnansweredQuestionHandler:
    """Gestionnaire des questions non répondues."""

    def __init__(self, rag_system=None):
        self.rag_system = rag_system
        self.unanswered_questions_file = Path("unanswered_questions.json")
        self.questions_data = self._load_questions()

    def _load_questions(self) -> Dict:
        """Charger les questions non répondues depuis le fichier JSON."""
        try:
            if self.unanswered_questions_file.exists():
                with open(self.unanswered_questions_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Erreur lors du chargement des questions: {e}")
        return {"questions": [], "patterns": []}

    def _save_questions(self):
        """Sauvegarder les questions non répondues."""
        try:
            with open(self.unanswered_questions_file, 'w', encoding='utf-8') as f:
                json.dump(self.questions_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des questions: {e}")

    def detect_unanswered_question(self, question: str, response: str) -> bool:
        """
        Détecte si une question n'a pas été correctement répondue.

        Args:
            question: La question posée
            response: La réponse reçue

        Returns:
            True si la question est considérée comme non répondue
        """
        # Patterns de réponses vides ou non satisfaisantes
        empty_patterns = [
            r"i don't have this information",
            r"i don't know",
            r"i'm not sure",
            r"i cannot find",
            r"not available",
            r"no information",
            r"insufficient data",
            r"pas d'information",
            r"je ne sais pas",
            r"je n'ai pas cette information",
            r"informations non disponibles"
        ]

        # Vérifier si la réponse contient des patterns de non-réponse
        for pattern in empty_patterns:
            if re.search(pattern, response.lower()):
                logger.info(f"Question non répondue détectée: {question[:100]}...")
                return True

        # Vérifier si la réponse est trop courte (< 50 mots)
        if len(response.split()) < 50 and not question.lower().startswith("how"):
            logger.warning(f"Réponse potentiellement insuffisante: {len(response.split())} mots")

        return False

    def handle_unanswered_question(self, question: str, response: str, context: Dict = None):
        """Gérer une question non répondue."""
        question_data = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "response": response,
            "context": context or {},
            "category": self._categorize_question(question)
        }

        # Ajouter à la liste des questions non répondues
        self.questions_data["questions"].append(question_data)

        # Créer une réponse de secours
        fallback_response = self._generate_fallback_response(question, context)

        # Sauvegarder les données
        self._save_questions()

        logger.info(f"Question non répondue enregistrée: {question[:50]}...")

        return fallback_response

    def _categorize_question(self, question: str) -> str:
        """Catégoriser la question pour une meilleure gestion."""
        question_lower = question.lower()

        categories = {
            "technical": ["python", "code", "programming", "bug", "error", "debug"],
            "documentation": ["doc", "documentation", "help", "how to", "guide"],
            "configuration": ["config", "setting", "setup", "install", "parameter"],
            "feature": ["feature", "functionality", "capability", "can i", "how do i"],
            "general": ["what is", "who", "when", "where", "why"]
        }

        for category, keywords in categories.items():
            if any(keyword in question_lower for keyword in keywords):
                return category

        return "other"

    def _generate_fallback_response(self, question: str, context: Dict = None) -> str:
        """Générer une réponse de secours."""
        base_responses = {
            "technical": "Je n'ai pas assez d'informations techniques pour répondre à cette question. Pourriez-vous fournir plus de contexte ou consulter la documentation officielle ?",
            "documentation": "Je ne trouve pas la documentation spécifique. Essayez de consulter le README.md ou la section 'Help' de l'application.",
            "configuration": "Les paramètres de configuration ne sont pas disponibles dans ma base de connaissances actuelle. Vérifiez le fichier config.py pour les options disponibles.",
            "feature": "Cette fonctionnalité n'est pas documentée dans mes connaissances actuelles. Essayez de l'explorer dans l'interface utilisateur ou consultez la documentation.",
            "general": "Je ne dispose pas de suffisamment d'informations pour répondre à cette question. Pourriez-vous reformuler ou poser une question plus spécifique ?",
            "other": "Je ne peux pas répondre à cette question avec les informations actuellement disponibles. Essayez de consulter la documentation ou demandez de l'aide."
        }

        category = self._categorize_question(question)

        # Ajouter des suggestions si un système RAG est disponible
        if self.rag_system:
            
        return base_responses[category]

    def get_common_unanswered_patterns(self) -> List[Dict]:
        """Identifier les motifs de questions non répondues fréquents."""
        if not self.questions_data["questions"]:
            return []

        # Analyser les motifs courants
        word_counts = {}
        for q_data in self.questions_data["questions"][-100:]:  # Dernières 100 questions
            words = re.findall(r'\b\w+\b', q_data["question"].lower())
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Retourner les mots/phrases les plus fréquents
        common_patterns = []
        for word, count in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
            if count > 2:  # Apparaît au moins 3 fois
                common_patterns.append({
                    "pattern": word,
                    "count": count,
                    "category": self._categorize_question(word)
                })

        return common_patterns

    def generate_improvement_report(self) -> str:
        """Générer un rapport d'amélioration basé sur les questions non répondues."""
        if not self.questions_data["questions"]:
            return "Aucune question non répondue enregistrée."

        total_questions = len(self.questions_data["questions"])
        categories = {}

        # Compter par catégorie
        for q_data in self.questions_data["questions"][-50:]:  # Dernières 50 questions
            cat = q_data.get("category", "other")
            categories[cat] = categories.get(cat, 0) + 1

        # Générer le rapport
        report = f"""
=== Rapport sur les Questions Non Répondues ===
Total des questions: {total_questions}

Répartition par catégorie:
"""
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            report += f"- {category}: {count} questions\n"

        # Ajouter les motifs courants
        patterns = self.get_common_unanswered_patterns()
        if patterns:
            report += f"\nMotifs fréquents dans les questions:\n"
            for pattern in patterns[:10]:
                report += f"- '{pattern['pattern']}': {pattern['count']} fois\n"

        # Suggestions d'amélioration
        report += f"""
\nSuggestions d'amélioration:
1. Mettre à jour les documents avec les informations manquantes
2. Améliorer les prompts pour les catégories fréquentes
3. Ajouter des réponses de secours spécifiques
4. Améliorer le système de recherche pour ces motifs
"""

        return report

class QuestionImprovementEngine:
    """Moteur d'amélioration des questions."""

    def __init__(self, unanswered_handler: UnansweredQuestionHandler):
        self.handler = unanswered_handler

    def suggest_improvements(self, question: str) -> Dict:
        """Suggérer des améliorations pour une question."""
        improvements = {
            "rephrase": self._suggest_rephrasing(question),
            "context": self._suggest_context(question),
            "specificity": self._suggest_specificity(question),
            "keywords": self._suggest_keywords(question)
        }
        return improvements

    def _suggest_rephrasing(self, question: str) -> str:
        """Suggérer une reformulation de la question."""
        if len(question.split()) < 10:
            return "Ajoutez plus de détails à votre question pour obtenir une meilleure réponse."

        if question.lower().startswith("what"):
            return "Essayez de commencer par 'How' ou 'Why' pour obtenir des réponses plus spécifiques."

        return "Votre question semble bien formulée."

    def _suggest_context(self, question: str) -> str:
        """Suggérer d'ajouter du contexte."""
        context_words = ["when", "where", "which version", "what platform"]

        if not any(word in question.lower() for word in context_words):
            return "Ajoutez du contexte comme la version du logiciel, le système d'exploitation, ou le cas d'usage spécifique."

        return "Le contexte semble suffisant."

    def _suggest_specificity(self, question: str) -> str:
        """Suggérer d'être plus spécifique."""
        general_words = ["problem", "issue", "error", "bug", "help"]

        if any(word in question.lower().split() for word in general_words):
            return "Sois plus spécifique en décrivant exactement le problème que tu rencontres."

        return "La question semble suffisamment spécifique."

    def _suggest_keywords(self, question: str) -> List[str]:
        """Suggérer des mots-clés pertinents."""
        keywords = []

        if "python" in question.lower():
            keywords.extend(["python3", "pip", "module", "library"])

        if "config" in question.lower():
            keywords.extend(["setting", "parameter", "option", "configuration"])

        if "error" in question.lower():
            keywords.extend(["exception", "traceback", "stack", "log"])

        return keywords[:5]  # Limiter à 5 suggestions

def main():
    """Fonction principale du script."""
    print("=== Système de Gestion des Questions Non Répondues ===")

    # Initialiser le handler
    handler = UnansweredQuestionHandler()

    # Exemple d'utilisation
    test_questions = [
        ("How do I install Python packages?", ""),
        ("What is the configuration file?", "I don't have this information"),
        ("How to fix a bug in my code?", ""),
        ("Can you help me?", "Not available")
    ]

    print("\n--- Analyse des questions de test ---")
    for question, response in test_questions:
        is_unanswered = handler.detect_unanswered_question(question, response)
        if is_unanswered:
            fallback = handler.handle_unanswered_question(question, response)
            print(f"\nQuestion: {question}")
            print(f"Fallback response: {fallback}")

    # Générer un rapport
    print("\n--- Rapport d'Amélioration ---")
    report = handler.generate_improvement_report()
    print(report)

    # Suggestions pour améliorer une question
    print("\n--- Suggestions d'Amélioration ---")
    improvement_engine = QuestionImprovementEngine(handler)
    test_question = "I have a problem with the app"
    suggestions = improvement_engine.suggest_improvements(test_question)

    print(f"Question: '{test_question}'")
    for suggestion_type, suggestion in suggestions.items():
        print(f"- {suggestion_type.title()}: {suggestion}")

if __name__ == "__main__":
    main()