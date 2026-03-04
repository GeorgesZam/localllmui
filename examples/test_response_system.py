#!/usr/bin/env python3
"""
Test script for the response improvement system.
Ce script démontre comment le système détecte et améliore les réponses insatisfaisantes.
"""

import sys
import os

# Ajouter le répertoire src au path pour importer les modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from response_handler import ConversationAwareResponseHandler

def test_response_handler():
    """Test du gestionnaire de réponses."""
    print("=== Test du Gestionnaire de Réponses ===\n")

    # Créer un handler avec un système RAG factice
    handler = ConversationAwareResponseHandler()

    # Cas de test
    test_cases = [
        {
            "question": "How do I install Python packages?",
            "response": "I don't have this information."
        },
        {
            "question": "What is the configuration file?",
            "response": "I don't know what configuration file you're referring to."
        },
        {
            "question": "How to fix a bug in my code?",
            "response": "Your code has a bug."
        },
        {
            "question": "Can you help me debug this?",
            "response": "I'd be happy to help! Please provide more details about the issue you're facing."
        }
    ]

    print("--- Traitement des réponses ---")
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest {i}:")
        print(f"Question: {case['question']}")
        print(f"Réponse originale: {case['response']}")

        # Traiter la réponse
        result = handler.process_response(
            case["question"],
            case["response"],
            {"conversation_id": "test_123", "has_rag": True}
        )

        print(f"Est insuffisante: {result['is_insufficient']}")
        if result['is_insufficient']:
            print("Suggestions améliorées:")
            if result['suggestions']:
                for key, suggestion in result['suggestions'].items():
                    print(f"  - {suggestion}")
            if result['follow_ups']:
                print("Questions de suivi:")
                for follow_up in result['follow_ups']:
                    print(f"  - {follow_up}")

    # Afficher les métriques
    print("\n--- Métriques de Qualité ---")
    metrics = handler.get_quality_metrics()
    print(f"Total réponses: {metrics['total']}")
    print(f"Réponses non répondues: {metrics['unanswered']}")
    print(f"Taux de réussite: {metrics['success_rate']:.1f}%")
    print(f"Longueur moyenne: {metrics['average_response_length']:.1f} mots")

def test_rag_integration():
    """Test de l'intégration avec RAG."""
    print("\n=== Test de l'Intégration RAG ===\n")

    # Créer un système RAG factice pour le test
    class MockRAG:
        def search(self, query, top_k=3, max_context_chars=500):
            # Retourner des documents factices
            return "", [
                {"title": "Python Installation Guide", "filename": "python_guide.md"},
                {"title": "Configuration Documentation", "filename": "config.md"}
            ]

    handler = ConversationAwareResponseHandler()

    # Test des tips RAG
    test_questions = [
        "How to install Python?",
        "What is the configuration?",
        "Hello, how are you?"
    ]

    print("--- Test des Tips RAG ---")
    for question in test_questions:
        should_show = handler.should_show_rag_tip(question)
        print(f"Question: '{question}'")
        print(f"Montrer tip RAG: {should_show}")

        # Obtenir des suggestions
        suggestions = handler.suggest_improvements_for_ui(question)
        if suggestions:
            print("Suggestions:")
            for key, suggestion in suggestions.items():
                print(f"  - {suggestion}")
        print()

def test_conversation_awareness():
    """Test de la conscience du contexte conversationnel."""
    print("\n=== Test de la Conscience Contextuelle ===\n")

    handler = ConversationAwareResponseHandler()

    # Simuler une conversation
    handler.set_conversation("conv_123")

    # Ajouter des messages précédents
    conversation_history = [
        {"question": "I'm having trouble with Python", "response": "What specific issue are you facing?"},
        {"question": "My code has a bug", "response": "I don't have enough information to help with that."},
    ]

    for entry in conversation_history:
        handler.add_to_conversation(entry["question"], entry["response"])

    # Obtenir le contexte
    context = handler.get_conversation_context("conv_123")
    print("Contexte de la conversation:")
    print(f"Longueur: {context['length']} échanges")
    print(f"Thèmes: {context['topics']}")
    print(f"Problèmes récurrents: {context['recurring_issues']}")
    if context['last_exchange']:
        print(f"Dernier échange: {context['last_exchange']['question']}")

def main():
    """Fonction principale."""
    print("=== Système d'Amélioration des Réponses ===\n")

    # Tester le système
    test_response_handler()
    test_rag_integration()
    test_conversation_awareness()

    print("\n=== Test Complet Terminé ===")
    print("\nCe système détecte automatiquement les réponses insuffisantes et :")
    print("1. Affiche des suggestions pour améliorer la question")
    print("2. Propose des questions de suivi pertinentes")
    print("3. Suggère d'utiliser RAG pour les questions techniques")
    print("4. Suit le contexte de la conversation")

if __name__ == "__main__":
    main()