# Knowledge Graph from IFC & GraphRAG

Ce projet permet d'ingérer un fichier IFC dans une base de données Neo4j, d'indexer les entités avec des embeddings OpenAI, puis d'interroger le graphe via une approche GraphRAG (Retrieval-Augmented Generation) pour répondre à des questions en langage naturel sur le modèle BIM.

## Fonctionnalités principales
- Ingestion complète d'un fichier IFC dans Neo4j (avec propriétés et relations)
- Génération d'embeddings vectoriels pour chaque entité via OpenAI
- Création d'un index vectoriel natif Neo4j pour la recherche sémantique
- Recherche des nœuds pertinents et de leurs chemins contextuels
- Génération de réponses en français à partir du contexte extrait

## Dépendances
- Python 3.9+
- Neo4j 5.12+ (avec support index vectoriel natif)
- OpenAI API
- ifcopenshell
- openai
- neo4j
- python-dotenv

## Installation
1. Clonez ce dépôt.
2. Installez les dépendances Python :
   ```bash
   pip install -r requirements.txt
   ```
3. Configurez un fichier `.env` à la racine avec vos identifiants :
   ```env
   OPENAI_API_KEY=sk-...
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PWD=motdepasse
   ```
4. Lancez Neo4j (5.12+) avec le plugin vectoriel activé.

## Utilisation
```bash
python Graphrag_pipeline.py chemin/vers/fichier.ifc "Votre question en français"
```
Exemple :
```bash
python Graphrag_pipeline.py basin-tessellation.ifc "Fais moi un resumé global de ce projet de construction"
python Graphrag_pipeline.py basin-tessellation.ifc "Où se trouve le lavabo R+1 ?"

```

## Notes
- L'ingestion du fichier IFC peut être longue selon la taille du modèle.
- Le script crée automatiquement l'index vectoriel si besoin.
- Les réponses sont générées en français et basées uniquement sur le graphe extrait.

