#!/bin/bash
# Script pour lancer l'application avec les dépendances correctes

echo "=== Démarrage de LocalLLMUI ==="

# Vérifier si on est dans le bon répertoire
if [ ! -f "pyproject.toml" ] || [ ! -d "src" ]; then
    echo "Erreur: exécutez ce script depuis le répertoire racine du projet"
    exit 1
fi

# Activer le virtual environment et lancer l'application
source .venv/bin/activate && \
PYTHONPATH=/Users/michelzam/localllmui/localllmui/src \
python src/main.py