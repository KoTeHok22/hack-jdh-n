import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


METALLURGY_KEYWORDS = {
    "material": [
        "пентландит", "халькопирит", "пирротин", "пирит", "халькозин", "борнит",
        "галенит", "сфалерит", "магнетит", "ильменит", "рутил",
        "руды", "концентрат", "хвосты", "шлам", "кек", "пульпа",
        "медно-никелевая руда", "сульфидные руды", "упорные руды",
        "золотосодержащая руда", "полиметаллическая руда",
    ],
    "element": [
        "никель", "медь", "железо", "сера", "золото", "серебро",
        "платина", "палладий", "кобальт", "цинк", "свинец",
        "ni", "cu", "fe", "s", "au", "ag", "pt", "pd",
        "элемент 28", "элемент 29",
        "благородные металлы", "цветные металлы",
    ],
    "process": [
        "флотация", "грохочение", "измельчение", "доизмельчение",
        "классификация", "сепарация", "магнитная сепарация",
        "гравитация", "гравитационное обогащение", "радиометрическая сепарация",
        "обжиг", "выщелачивание", "цианирование", "амальгамация",
        "концентрация", "обогащение", "дробление", "сортировка",
        "пенная флотация", "масляная агломерация", "коагуляция",
    ],
    "property": [
        "извлечение", "содержание", "крупность", "потери", "масса",
        "степень раскрытия", "гидрофильность", "гидрофобность",
        "смачиваемость", "плотность", "твердость",
        "селективность", "кондиция", "качество",
    ],
    "equipment": [
        "мельница", "флотомашина", "гидроциклон", "грохот",
        "сепаратор", "классификатор", "дробилка", "конвейер",
        "бункер", "насос", "чаща", "колесо",
        "шаровая мельница", "стержневая мельница",
        "флотационная машина", "магнитный сепаратор",
    ]
}


def _extract_keywords_from_text(text: str) -> tuple[List[Dict], List[Dict]]:
    """Простое извлечение сущностей на основе ключевых слов."""
    entities = []
    found = set()
    
    text_lower = text.lower()
    
    for entity_type, keywords in METALLURGY_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower and keyword.lower() not in found:
                found.add(keyword.lower())
                entities.append({
                    "name": keyword,
                    "type": entity_type,
                })
    
    return entities, []


def _build_graph_from_hypotheses(
    problem_statement: str,
    hypotheses: List[Dict],
) -> Dict[str, Any]:
    """Построение графа из проблемы и гипотез без LLM."""
    nodes_map = {}
    edges = []
    
    full_text = problem_statement
    for h in hypotheses:
        full_text += f"\n{h.get('statement', '')}\n{h.get('mechanism', '')}\n"
    
    entities, _ = _extract_keywords_from_text(full_text)
    
    for entity in entities:
        name = entity["name"].strip().lower()
        if not name:
            continue
        
        entity_type = entity.get("type", "property")
        node_id = f"{entity_type}_{name}"
        
        if node_id not in nodes_map:
            nodes_map[node_id] = {
                "data": {
                    "id": node_id,
                    "label": entity["name"],
                    "type": entity_type,
                    "color": _get_entity_color(entity_type),
                }
            }
    
    problem_node_id = "problem_root"
    nodes_map[problem_node_id] = {
        "data": {
            "id": problem_node_id,
            "label": problem_statement[:80] + "..." if len(problem_statement) > 80 else problem_statement,
            "type": "problem",
            "color": "#ffffff",
        }
    }
    
    entity_node_ids = list(nodes_map.keys())
    for eid in entity_node_ids:
        if eid != problem_node_id:
            edges.append({
                "data": {
                    "id": f"{problem_node_id}_{eid}_related",
                    "source": problem_node_id,
                    "target": eid,
                    "type": "related",
                }
            })
    
    for i, hyp in enumerate(hypotheses):
        h_id = f"hypothesis_{i}"
        h_statement = hyp.get("statement", "")
        h_label = h_statement[:60] + "..." if len(h_statement) > 60 else h_statement
        
        nodes_map[h_id] = {
            "data": {
                "id": h_id,
                "label": f"H{i+1}: {h_label}",
                "type": "hypothesis",
                "color": "#ff4444",
            }
        }
        
        edges.append({
            "data": {
                "id": f"{problem_node_id}_{h_id}_generates",
                "source": problem_node_id,
                "target": h_id,
                "type": "generates",
            }
        })
        
        hyp_text = f"{h_statement} {hyp.get('mechanism', '')}".lower()
        for entity in entities:
            if entity["name"].lower() in hyp_text:
                entity_type = entity.get("type", "property")
                entity_node_id = f"{entity_type}_{entity['name'].lower()}"
                if entity_node_id in nodes_map:
                    edges.append({
                        "data": {
                            "id": f"{h_id}_{entity_node_id}_proposes",
                            "source": h_id,
                            "target": entity_node_id,
                            "type": "proposes",
                        }
                    })
    
    nodes = list(nodes_map.values())
    
    type_counts = {}
    for node in nodes:
        t = node["data"]["type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    
    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": type_counts,
        }
    }


def _get_entity_color(entity_type: str) -> str:
    colors = {
        "material": "#0077bc",
        "element": "#009866",
        "process": "#f59e0b",
        "property": "#8b5cf6",
        "equipment": "#ec4899",
        "problem": "#ffffff",
        "hypothesis": "#ff4444",
    }
    return colors.get(entity_type, "#888888")
