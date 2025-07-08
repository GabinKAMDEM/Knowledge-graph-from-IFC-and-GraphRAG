# neo4j_graphrag_pipeline.py
"""
Pipeline complet : ingestion d'un fichier IFC dans Neo4j + interrogation GraphRAG
(compatible Neo4j 5.12+ avec index vectoriel natif)

Correctif : évite l'erreur
```
neo4j.exceptions.CypherTypeError: Property values can only be of primitive types ... Encountered: Map{}
```
— on sérialise désormais le dictionnaire de propriétés en **JSON string**, autorisé par Neo4j (type `String`).

Dépendances :
```bash
pip install ifcopenshell neo4j openai python-dotenv
```

Usage :
```bash
python neo4j_graphrag_pipeline.py model.ifc "Où se trouve le lavabo R+1 ?"
```
"""

from __future__ import annotations
import os
import sys
import json
from typing import List, Dict, Tuple, Set, Optional, Any
from uuid import uuid4
import re

import ifcopenshell
from neo4j import GraphDatabase
import openai
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# 0. ENV & CLIENTS
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PWD = os.getenv("NEO4J_PWD", "password")
openai.api_key = OPENAI_API_KEY

# Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))

class IfcNode:
    id: str | None
    uuid: str | None
    label: str
    labels: Set[str]
    properties: Dict[str, Any]

    def __init__(self, entity_id: str | None, class_name: str, super_classes: Optional[List[str]] = None, properties: Optional[Dict[str, Any]] = None):
        if entity_id:
            self.id = entity_id
            self.uuid = None
        else:
            self.uuid = str(uuid4())
            self.id = self.uuid

        self.label = class_name
        self.labels = {class_name}
        if super_classes:
            self.labels.update(super_classes)

        self.properties = {
            '__instance_of': class_name,
            '__step_file_id': self.id,
            '__step_file_uuid': self.uuid
        }
        if properties:
            self.properties.update(properties)

def get_ifc_hierarchy(entity) -> List[str]:
    """Récupère la hiérarchie des classes IFC de manière sécurisée."""
    hierarchy = []
    try:
        # Commence par la classe actuelle
        current_class = entity.is_a()
        hierarchy.append(current_class)
        
        # Remonte la hiérarchie en utilisant la méthode correcte
        while True:
            try:
                # Utilise la méthode correcte pour obtenir le type parent
                parent = entity.wrapped_data.type().supertype()
                if parent is None:
                    break
                hierarchy.append(parent.name())
                entity = parent
            except (AttributeError, TypeError):
                break
    except Exception as e:
        print(f"Attention: Impossible de récupérer la hiérarchie pour {entity}: {str(e)}")
    
    return hierarchy

def get_ifc_properties(entity) -> Dict[str, Any]:
    """Récupère les propriétés IFC de manière sécurisée."""
    properties = {}
    try:
        # Propriétés de base
        if hasattr(entity, "GlobalId"):
            properties["GlobalId"] = entity.GlobalId
        if hasattr(entity, "Name"):
            properties["Name"] = str(entity.Name) if entity.Name else None
        if hasattr(entity, "Description"):
            properties["Description"] = str(entity.Description) if entity.Description else None

        # Propriétés définies via Pset
        for rel in getattr(entity, "IsDefinedBy", []):
            try:
                if rel.is_a("IfcRelDefinesByProperties"):
                    pset = rel.RelatingPropertyDefinition
                    if hasattr(pset, "Name"):
                        pset_name = pset.Name
                        for prop in getattr(pset, "HasProperties", []):
                            try:
                                if hasattr(prop, "Name") and hasattr(prop, "NominalValue"):
                                    name = f"{pset_name}.{prop.Name}"
                                    if hasattr(prop.NominalValue, "wrappedValue"):
                                        val = prop.NominalValue.wrappedValue
                                    else:
                                        val = str(prop.NominalValue)
                                    properties[name] = val
                            except Exception as e:
                                print(f"Attention: Erreur lors de l'extraction de la propriété {prop}: {str(e)}")
            except Exception as e:
                print(f"Attention: Erreur lors du traitement du Pset {rel}: {str(e)}")

        # Propriétés de matériau
        if hasattr(entity, "HasAssociations"):
            for assoc in entity.HasAssociations:
                try:
                    if assoc.is_a("IfcRelAssociatesMaterial"):
                        material = assoc.RelatingMaterial
                        if material:
                            if material.is_a("IfcMaterial"):
                                properties["Material"] = material.Name
                            elif material.is_a("IfcMaterialLayerSet"):
                                for layer in material.MaterialLayers:
                                    if hasattr(layer, "Material") and layer.Material:
                                        properties["Material"] = layer.Material.Name
                except Exception as e:
                    print(f"Attention: Erreur lors de l'extraction du matériau: {str(e)}")

    except Exception as e:
        print(f"Attention: Erreur lors de l'extraction des propriétés pour {entity}: {str(e)}")
    
    return properties

def create_ifc_node(ifc_entity, ifc_file, hierarchy: bool = True) -> IfcNode:
    """Crée un nœud IFC avec ses propriétés et sa hiérarchie."""
    try:
        # Récupère l'ID de manière sécurisée
        entity_id = None
        try:
            if hasattr(ifc_entity, "GlobalId"):
                entity_id = ifc_entity.GlobalId
        except Exception as e:
            print(f"Attention: Impossible de récupérer l'ID pour {ifc_entity}: {str(e)}")

        # Récupère le nom de la classe
        try:
            class_name = ifc_entity.is_a()
        except Exception as e:
            print(f"Attention: Impossible de récupérer le type pour {ifc_entity}: {str(e)}")
            class_name = "IfcRoot"

        # Récupère la hiérarchie si demandée
        super_classes = []
        if hierarchy:
            super_classes = get_ifc_hierarchy(ifc_entity)

        # Récupère les propriétés
        properties = get_ifc_properties(ifc_entity)

        return IfcNode(entity_id, class_name, super_classes, properties)
    except Exception as e:
        print(f"Erreur critique lors de la création du nœud pour {ifc_entity}: {str(e)}")
        # Retourne un nœud minimal en cas d'erreur
        return IfcNode(None, "IfcRoot", [], {"error": str(e)})

def upsert_node(tx, node: IfcNode):
    """Crée ou met à jour un nœud dans Neo4j."""
    # Crée d'abord le nœud avec son label principal
    query = """
    MERGE (n:Ifc {id: $id})
    SET n += $properties
    """
    tx.run(query, 
           id=node.id,
           properties=node.properties)
    
    # Ajoute les labels supplémentaires un par un
    for label in node.labels:
        if label != "Ifc":  # Skip le label principal déjà ajouté
            query = f"""
            MATCH (n {{id: $id}})
            SET n:{label}
            """
            tx.run(query, id=node.id)

def sanitize_rel_type(rel_type: str) -> str:
    """Nettoie le nom du type de relation pour qu'il soit valide en Cypher."""
    # Remplace les caractères non autorisés par des underscores
    # Garde uniquement les lettres, chiffres et underscores
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', rel_type)
    # S'assure que le nom commence par une lettre
    if not clean[0].isalpha():
        clean = 'REL_' + clean
    return clean

def create_relationship(tx, from_id: str, to_id: str, rel_type: str):
    """Crée une relation entre deux nœuds."""
    # Nettoie le type de relation
    safe_rel_type = sanitize_rel_type(rel_type)
    
    # Crée la relation avec un type nettoyé
    query = f"""
    MATCH (from:Ifc {{id: $from_id}})
    MATCH (to:Ifc {{id: $to_id}})
    MERGE (from)-[r:{safe_rel_type}]->(to)
    SET r.original_type = $original_type
    """
    tx.run(query, 
           from_id=from_id, 
           to_id=to_id, 
           original_type=rel_type)  # Garde le type original comme propriété

def embed(text: str) -> List[float]:
    """Génère un embedding vectoriel pour un texte."""
    client = openai.OpenAI()
    resp = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return resp.data[0].embedding

def add_embeddings(tx, node: IfcNode):
    """Ajoute un embedding vectoriel à un nœud."""
    # Crée un texte descriptif à partir des propriétés du nœud
    text = f"{node.label} {' '.join(str(v) for v in node.properties.values())}"
    emb = embed(text)
    tx.run(
        "MATCH (n:Ifc {id: $id}) SET n.embedding = $emb",
        id=node.id,
        emb=emb
    )

def create_vector_index(tx):
    """Crée l'index vectoriel pour la recherche sémantique."""
    result = tx.run("SHOW INDEXES")
    indexes = [record["name"] for record in result if record["name"] == "ifc_emb"]
    
    if not indexes:
        tx.run(
            "CALL db.index.vector.createNodeIndex('ifc_emb', 'Ifc', 'embedding', 1536, 'cosine')"
        )

def clean_database_schema(tx):
    """Nettoie les éléments de schéma (index)."""
    tx.run("DROP INDEX ifc_emb IF EXISTS")

def clean_database_data(tx):
    """Nettoie les données (embeddings)."""
    tx.run("MATCH (n:Ifc) REMOVE n.embedding")

def process_ifc_entity(tx, ifc_entity, ifc_file):
    """Traite une entité IFC et crée ses nœuds et relations."""
    try:
        # Crée le nœud principal
        node = create_ifc_node(ifc_entity, ifc_file)
        if node.id:  # Ne traite que les nœuds avec un ID valide
            try:
                upsert_node(tx, node)
                # Ajoute l'embedding seulement si on a des propriétés significatives
                if any(v for v in node.properties.values() if v):
                    add_embeddings(tx, node)
            except Exception as e:
                print(f"Attention: Erreur lors de la création du nœud {node.id}: {str(e)}")
                return

            # Traite les relations directes
            try:
                for i in range(ifc_entity.__len__()):
                    if not ifc_entity[i]:
                        continue

                    try:
                        arg_type = ifc_entity.wrapped_data.get_argument_type(i)
                        arg_name = ifc_entity.wrapped_data.get_argument_name(i)

                        if arg_type == 'ENTITY INSTANCE':
                            if ifc_entity[i].is_a() == 'IfcOwnerHistory' and ifc_entity.is_a() != 'IfcProject':
                                continue

                            sub_node = create_ifc_node(ifc_entity[i], ifc_file)
                            if sub_node.id:  # Ne traite que les sous-nœuds avec un ID valide
                                try:
                                    upsert_node(tx, sub_node)
                                    if any(v for v in sub_node.properties.values() if v):
                                        add_embeddings(tx, sub_node)
                                    create_relationship(tx, node.id, sub_node.id, arg_name)
                                except Exception as e:
                                    print(f"Attention: Erreur lors du traitement du sous-nœud {sub_node.id}: {str(e)}")

                        elif arg_type == 'AGGREGATE OF ENTITY INSTANCE':
                            for sub_entity in ifc_entity[i]:
                                sub_node = create_ifc_node(sub_entity, ifc_file)
                                if sub_node.id:  # Ne traite que les sous-nœuds avec un ID valide
                                    try:
                                        upsert_node(tx, sub_node)
                                        if any(v for v in sub_node.properties.values() if v):
                                            add_embeddings(tx, sub_node)
                                        create_relationship(tx, node.id, sub_node.id, arg_name)
                                    except Exception as e:
                                        print(f"Attention: Erreur lors du traitement du sous-nœud {sub_node.id}: {str(e)}")
                    except Exception as e:
                        print(f"Attention: Erreur lors du traitement de l'attribut {i} pour {ifc_entity}: {str(e)}")

            except Exception as e:
                print(f"Attention: Erreur lors du traitement des relations pour {ifc_entity}: {str(e)}")

    except Exception as e:
        print(f"Erreur critique lors du traitement de l'entité {ifc_entity}: {str(e)}")

def ingest(ifc_path: str):
    """Ingère un fichier IFC dans Neo4j."""
    print(f"\n[Ingestion] Début de l'ingestion du fichier {ifc_path}")
    ifc_file = ifcopenshell.open(ifc_path)
    
    with driver.session() as sess:
        # Nettoie d'abord le schéma
        print("[Ingestion] Nettoyage de la base de données...")
        sess.execute_write(clean_database_schema)
        sess.execute_write(clean_database_data)
        
        # Compte le nombre d'entités
        entity_count = len(ifc_file.wrapped_data.entity_names())
        print(f"[Ingestion] Nombre total d'entités IFC à traiter : {entity_count}")
        
        # Traite chaque entité
        processed_count = 0
        for entity_id in ifc_file.wrapped_data.entity_names():
            entity = ifc_file.by_id(entity_id)
            sess.execute_write(process_ifc_entity, entity, ifc_file)
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"[Ingestion] Progression : {processed_count}/{entity_count} entités traitées")
        
        # Crée l'index vectoriel
        print("[Ingestion] Création de l'index vectoriel...")
        sess.execute_write(create_vector_index)
        
        # Vérifie le nombre de nœuds créés
        result = sess.run("MATCH (n:Ifc) RETURN count(n) as count").single()
        print(f"[Ingestion] Nombre de nœuds créés dans Neo4j : {result['count']}")
        
        # Vérifie le nombre de nœuds avec des embeddings
        result = sess.run("MATCH (n:Ifc) WHERE n.embedding IS NOT NULL RETURN count(n) as count").single()
        print(f"[Ingestion] Nombre de nœuds avec embeddings : {result['count']}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. GRAPH RAG : Retrieval
# ─────────────────────────────────────────────────────────────────────────────

class Neo4jPathSerializer:
    def __init__(self, path_data):
        """Sérialise un chemin Neo4j en format JSON.
        
        Args:
            path_data: Soit un objet Path Neo4j, soit un dictionnaire contenant:
                      - start: nœud de départ
                      - rels: liste de relations avec leurs propriétés
                      - end: nœud d'arrivée
        """
        if hasattr(path_data, 'nodes'):  # Ancien format (objet Path)
            self.nodes = [
                {"id": n["id"], "type": n["type"], "name": n["name"]}
                for n in path_data.nodes
            ]
            self.rels = [rel.type for rel in path_data.relationships]
        else:  # Nouveau format (dictionnaire)
            self.nodes = []
            self.rels = []
            seen_nodes = set()  # Pour éviter les doublons
            
            # Traite le nœud de départ
            start_node = path_data["start"]
            if start_node["id"] not in seen_nodes:
                self.nodes.append({
                    "id": start_node["id"],
                    "type": start_node.get("type", ""),
                    "name": start_node.get("name", ""),
                    "props": start_node.get("props", {})
                })
                seen_nodes.add(start_node["id"])
            
            # Traite les relations
            for rel in path_data["rels"]:
                # Utilise le type original s'il existe, sinon le type de la relation
                rel_type = rel.get("original_type", rel.get("type", ""))
                self.rels.append(rel_type)
            
            # Traite le nœud d'arrivée
            end_node = path_data["end"]
            if end_node["id"] not in seen_nodes:
                self.nodes.append({
                    "id": end_node["id"],
                    "type": end_node.get("type", ""),
                    "name": end_node.get("name", ""),
                    "props": end_node.get("props", {})
                })
                seen_nodes.add(end_node["id"])

    def to_dict(self):
        """Convertit l'objet en dictionnaire pour la sérialisation JSON."""
        return {
            "nodes": self.nodes,
            "rels": self.rels
        }

    def __repr__(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)

def retrieve(question: str, k: int = 3) -> Tuple[str, List[Dict]]:
    """Récupère les nœuds pertinents et leurs chemins associés."""
    print(f"\n[Recherche] Question : {question}")
    embedding = embed(question)
    print(f"[Recherche] Embedding généré (dimensions : {len(embedding)})")
    
    with driver.session() as sess:
        # Vérifie d'abord si l'index existe
        result = sess.run("SHOW INDEXES").data()
        indexes = [r["name"] for r in result if r["name"] == "ifc_emb"]
        if not indexes:
            print("[Recherche] ERREUR : L'index vectoriel 'ifc_emb' n'existe pas !")
            return json.dumps({"hits": [], "paths": [], "error": "Index vectoriel manquant"}), []
        
        # Vérifie le nombre de nœuds avec des embeddings
        result = sess.run("MATCH (n:Ifc) WHERE n.embedding IS NOT NULL RETURN count(n) as count").single()
        node_count = result["count"]
        print(f"[Recherche] Nombre de nœuds avec embeddings : {node_count}")
        
        if node_count == 0:
            print("[Recherche] ERREUR : Aucun nœud n'a d'embedding !")
            return json.dumps({"hits": [], "paths": [], "error": "Aucun embedding trouvé"}), []
        
        # Recherche vectorielle
        print("[Recherche] Exécution de la recherche vectorielle...")
        cypher_vec = """
        CALL db.index.vector.queryNodes('ifc_emb', $k, $e) 
        YIELD node, score
        RETURN node.id AS gid, 
               node.type AS type, 
               node.name AS name, 
               node.props AS props, 
               score
        """
        hits = sess.run(cypher_vec, k=k, e=embedding).data()
        print(f"[Recherche] Nombre de résultats trouvés : {len(hits)}")
        
        if not hits:
            print("[Recherche] Aucun résultat trouvé pour la recherche vectorielle")
            return json.dumps({"hits": [], "paths": []}, ensure_ascii=False), []
        
        # Affiche les scores des résultats
        print("\n[Recherche] Scores des résultats :")
        for hit in hits:
            print(f"- {hit['type']} {hit['name']} (score: {hit['score']:.4f})")
        
        # Récupère les chemins
        gids = [h["gid"] for h in hits]
        print(f"\n[Recherche] Recherche des chemins pour {len(gids)} nœuds...")
        cypher_ctx = """
        MATCH p=(n:Ifc)-[*1..2]-(m:Ifc) 
        WHERE n.id IN $gids 
        WITH n, m, relationships(p) as rels
        RETURN {
            start: n,
            rels: [rel in rels | {
                type: type(rel),
                original_type: rel.original_type,
                properties: properties(rel)
            }],
            end: m
        } as segment
        LIMIT 100
        """
        paths = sess.run(cypher_ctx, gids=gids).data()
        print(f"[Recherche] Nombre de chemins trouvés : {len(paths)}")
        
    context = {
        "hits": hits,
        "paths": [Neo4jPathSerializer(p["segment"]).to_dict() for p in paths]
    }
    return json.dumps(context, ensure_ascii=False), hits

# ─────────────────────────────────────────────────────────────────────────────
# 3. Génération LLM
# ─────────────────────────────────────────────────────────────────────────────

def answer(question: str, context_json: str) -> str:
    client = openai.OpenAI()
    prompt = f"""
    Tu es un assistant BIM. Réponds à la question en te basant **uniquement** sur le contexte JSON suivant :
    ```json
    {context_json}
    ```
    Question : {question}
    Réponse (français) :
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ─────────────────────────────────────────────────────────────────────────────
# 4. CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage : python Graphrag_pipeline.py <fichier.ifc> <question>")
        sys.exit(1)
    ifc_file, question = sys.argv[1], " ".join(sys.argv[2:])
    #ingest(ifc_file)
    ctx, hits = retrieve(question)
    print("[Retrieval] Noeuds pertinents :", hits)
    print("\n[Answer]\n", answer(question, ctx))
