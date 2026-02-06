from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import textwrap
from pydantic import BaseModel


# -----------------------------
# Formatting helpers
# -----------------------------

def _format_excerpts(excerpts: List[str], max_chars_each: int = 700) -> str:
    blocks = []
    for idx, ex in enumerate(excerpts):
        txt = ex.strip()
        if len(txt) > max_chars_each:
            txt = txt[: max_chars_each].rstrip() + "…"
        blocks.append(f"- [{idx+1}] {txt}")
    return "\n".join(blocks) if blocks else "(no excerpts provided)"


def _format_graph_option(opt: GraphOption) -> str:
    nodes_str = ", ".join(opt.nodes) if opt.nodes else "(none)"
    edges_lines = []
    for e in opt.edges:
        edges_lines.append(f"  - {e.source} -[{e.relation_type}]-> {e.target}")
    edges_str = "\n".join(edges_lines) if edges_lines else "  (none)"
    return textwrap.dedent(f"""
        {opt.option_id}
        Nodes: {nodes_str}
        Edges:
        {edges_str}
    """).strip()


def _json_output_contract() -> str:
    return textwrap.dedent("""
        Output format (STRICT JSON, no markdown, no trailing commentary):
        {
          "score": <integer in the required range>,
          "rationale": "<2-6 sentences; must reference excerpt_ids as evidence>",
          "evidence": ["<excerpt_id>", "..."],
          "notes": ["<optional brief note>", "..."]
        }
    """).strip()

class Output(BaseModel):
    score: int
    rationale: str
    evidence: List[str]
    notes: List[str]



# -----------------------------
# Ordinal 0–2 rubric (Metric 1)
# -----------------------------

ORDINAL_0_2_NODE_RUBRIC = textwrap.dedent("""
    Use the following 0–2 ordinal scale strictly (Concept Node significance/validity):

    Definitions:
    - "Significant/meaningful concept" = a critical educational concept that students should
      be taught and be able to explain/use in this course (core topic, method, principle,
      model, theorem, algorithm, technique, framework, key term).
    - "Not meaningful for the course" includes logistical/admin/metadata items that do not
      represent course content knowledge.

    Clear examples:
    - Meaningful (YES / likely 2 if supported): "recursion", "merge sort", "Bayes' theorem",
      "photosynthesis", "supply and demand", "gradient descent", "constitutional amendments".
    - Not meaningful (NO / likely 0 even if mentioned): instructor/TA names, office hours,
      due dates, grading policy, course number, Zoom link, classroom location, required textbook ISBN,
      "homework submission", "attendance", "Canvas", "midterm".

    Scoring:
    0 = Invalid OR not a course-content concept (logistics/metadata), OR clearly insignificant or unrelated to course learning goals.
    1 = Plausible course-content concept but weakly supported, too vague, overly broad, or ambiguous given excerpts.
        (Use 1 when you cannot confirm significance from the excerpts.)
    2 = Clearly a valid course-content concept AND clearly significant for the course, with strong excerpt support.

    Evidence policy:
    - Base your score ONLY on the provided excerpts and the node label.
    - If evidence is insufficient/ambiguous, score 1 and state what is missing.
    - Cite evidence by excerpt_id (avoid long quotations).
""").strip()

ORDINAL_0_2_TRIPLET_RUBRIC = textwrap.dedent("""
    Use the following 0–2 ordinal scale strictly (Directed, typed edge accuracy):

    You must judge TWO things:
    (1) Whether A and B are directly related as course concepts, AND
    (2) Whether the relation TYPE AND DIRECTION are correct.

    If A and B should NOT be directly related, then the proposed relationship is INCORRECT (e.g., "mergesort", "machine learning" should not be directly related).

    Relation semantics (direction matters):
    - depends_on: Comprehending A requires understanding B; B is a prerequisite of A. Students should learn/understand B BEFORE A.
      (Read as: A depends_on B.)
    - part_of: A is a subtopic/component of B. B contains/organizes A.
      (Read as: A is part_of B.)

    Clear examples (directional):
    - Correct depends_on:
      A="merge sort" depends_on B="recursion"  ✅ (merge sort relies on recursion ideas)
      A="backpropagation" depends_on B="chain rule" ✅
    - Incorrect depends_on (reversed):
      A="recursion" depends_on B="merge sort" ❌ (direction reversed; recursion is more fundamental)
    
    - Correct part_of:
      A="merge sort" part_of B="sorting algorithms" and relation part_of where A part_of B ✅
      A="mitochondria" with B="cell" and relation part_of where A part_of B ✅
    - Incorrect part_of (reversed):
      A="sorting algorithms" part_of B="merge sort" ❌

    Scoring:
    0 = No: the proposed direct relationship is wrong OR not supported by excerpts.
    1 = Somewhat: A and B are related, but the type and/or the direction is wrong or unclear from evidence.
    2 = Yes: A and B are directly related AND the type AND direction match the excerpts.

    Evidence policy:
    - Base your score ONLY on the provided excerpts and the proposed edge.
    - If evidence is insufficient/ambiguous, score 1 and state what is missing.
    - Cite evidence by excerpt_id (avoid long quotations).
""").strip()



# -----------------------------
# Likert 1–5 rubric (Metric 1)
# -----------------------------

LIKERT_1_5_RUBRIC_COMMON = textwrap.dedent("""
    Use the following Likert scale strictly:

    1 = Strongly prefer / Very poor / Not at all (clear evidence it is worse; major issues)
    2 = Somewhat prefer / Poor (noticeably worse; multiple issues)
    3 = No preference / Mixed / Adequate (roughly equal; or unclear from evidence)
    4 = Somewhat prefer / Good (noticeably better; minor issues)
    5 = Strongly prefer / Excellent / Extremely (clear evidence it is better; minimal issues)

    Evidence policy:
    - Base your score ONLY on the provided excerpts and the provided nodes/edges.
    - If evidence is insufficient or ambiguous, score 3 and say what is missing.
    - Cite evidence by excerpt_id (do not quote long passages).
""").strip()


# -----------------------------
# Metric (1) — Ordinal 0–2 prompts
# -----------------------------

def prompt_m1_concept_node_validity_ordinal(
    node_label: str,
    excerpts: List[str],
    course_name: str
) -> str:
    return textwrap.dedent(f"""
You are evaluating a single concept node extracted for a course knowledge graph.

Course Title: {course_name}

Goal:
Decide whether the node is a SIGNIFICANT course-content concept:
something students should be taught and should understand/use in this course.

Important:
- "Significant concept" means key educational concepts (topic, method, principle, theorem,
    algorithm, framework, key technical term) for a course.
- It does NOT mean logistics/admin/metadata (e.g., instructor name, due dates, office hours,
    grading policy, LMS/Canvas, Zoom link).
- It does NOT mean educational concepts that would typically fall under a different course/topic (e.g., "Constitution", mentioned in an example of text parsing, would NOT be a significant concept for an NLP course.)

Concept Node:
- A = "{node_label}"

{ORDINAL_0_2_NODE_RUBRIC}

Instructor-material excerpts (evidence base):
{_format_excerpts(excerpts)}

Requirements for your answer:
- Output STRICT JSON only (no markdown).
- Rationale must cite excerpt_id(s) as evidence (e.g., "[1]", "[3]").
- If evidence is insufficient or ambiguous, score 1 and say what is missing.

        {_json_output_contract()}
    """).strip()



def prompt_m1_concept_triplet_accuracy_ordinal(
    edge: dict,
    excerpts: List[str],
    course_name: str
) -> str:
    """
    Metric (1) Concept Triplet question (0–2):
    "Is the edge type and direction between node A and B an accurate reflection...?"
    """
    return textwrap.dedent(f"""
        You are evaluating a single directed typed edge (concept triplet) in a course knowledge graph.

        Task:
        Score whether the edge type and direction accurately reflect the conceptual relationship
        between the two concepts, based strictly on the instructor-provided excerpts.

        Course Title: {course_name}

        Concept Triplet:
        - A = "{edge['source']}"
        - relation = "{edge['relation_type']}"  (allowed: depends_on, part_of)
        - B = "{edge['target']}"
        Interpreting relation types:
        - depends_on: B is a prerequisite of A (B should be learned before A)
        - part_of:    B is a subtopic/component of A (A contains/organizes B)

        {ORDINAL_0_2_TRIPLET_RUBRIC}

        Instructor-material excerpts (evidence base):
        { _format_excerpts(excerpts) }

        { _json_output_contract() }
    """).strip()


# -----------------------------
# Metric (1) — Likert 1–5 prompts (pairwise)
# -----------------------------

def prompt_compare_major_concepts(option1: GraphOption, option2: GraphOption, excerpts: List[Excerpt]) -> str:
    return textwrap.dedent(f"""
        You are evaluating two candidate knowledge-graph subgraphs derived from instructor-provided course material.

        Task:
        Score which option better reflects the MAJOR concepts that are important for this portion of the course.

        Scale interpretation for this item:
        - 1 = Strongly prefer Option #1
        - 3 = No preference / roughly equal / insufficient evidence
        - 5 = Strongly prefer Option #2

        {LIKERT_1_5_RUBRIC_COMMON}

        Evaluation criteria (apply all):
        - Coverage of key concepts emphasized in excerpts (breadth and salience).
        - Inclusion of central concepts vs peripheral trivia.
        - Concept labels match the material’s terminology and level (avoid overly vague or off-scope nodes).
        - Penalize options that omit clearly emphasized concepts or include multiple irrelevant concepts.

        Inputs:
        { _format_graph_option(option1) }

        { _format_graph_option(option2) }

        Instructor-material excerpts (evidence base):
        { _format_excerpts(excerpts) }

        { _json_output_contract() }
    """).strip()


def prompt_compare_confidence_completeness_relevance(option1: GraphOption, option2: GraphOption, excerpts: List[Excerpt]) -> str:
    return textwrap.dedent(f"""
        You are evaluating two candidate knowledge-graph subgraphs derived from instructor-provided course material.

        Task:
        Score which option you have MORE CONFIDENCE IN regarding:
        (a) not omitting crucial concepts, and (b) not including irrelevant concepts.

        Scale interpretation for this item:
        - 1 = Strong confidence in Option #1
        - 3 = No preference / roughly equal / insufficient evidence
        - 5 = Strong confidence in Option #2

        {LIKERT_1_5_RUBRIC_COMMON}

        Evaluation criteria:
        - Crucial-concept recall: are the clearly necessary concepts (per excerpts) present?
        - Precision: are there concepts that appear unsupported, tangential, or mismatched to the material?
        - Balance: prefer the option that is both more complete AND less noisy.
        - If one option adds many nodes, do not reward quantity unless excerpts support them.

        Inputs:
        { _format_graph_option(option1) }

        { _format_graph_option(option2) }

        Instructor-material excerpts (evidence base):
        { _format_excerpts(excerpts) }

        { _json_output_contract() }
    """).strip()


def prompt_compare_relationship_accuracy(option1: GraphOption, option2: GraphOption, excerpts: List[Excerpt]) -> str:
    return textwrap.dedent(f"""
        You are evaluating two candidate knowledge-graph subgraphs derived from instructor-provided course material.

        Task:
        Score which option more accurately represents the RELATIONSHIPS between concepts:
        - depends_on: prerequisite (A must be understood before B)
        - part_of: subtopic (B is a component/subtopic of A)

        Scale interpretation for this item:
        - 1 = Option #1 is much more accurate
        - 3 = No preference / roughly equal / insufficient evidence
        - 5 = Option #2 is much more accurate

        {LIKERT_1_5_RUBRIC_COMMON}

        Evaluation criteria:
        - Direction correctness: prerequisite flow and containment direction match the excerpts.
        - Type correctness: depends_on vs part_of is chosen appropriately.
        - Structural plausibility: avoid cycles of prerequisites unless clearly evidenced.
        - Penalize edges that are unsupported by excerpts or contradict them.

        Inputs:
        { _format_graph_option(option1) }

        { _format_graph_option(option2) }

        Instructor-material excerpts (evidence base):
        { _format_excerpts(excerpts) }

        { _json_output_contract() }
    """).strip()


def prompt_compare_granularity(option1: GraphOption, option2: GraphOption, excerpts: List[Excerpt]) -> str:
    return textwrap.dedent(f"""
        You are evaluating two candidate knowledge-graph subgraphs derived from instructor-provided course material.

        Task:
        Score which option has a more appropriate level of CONCEPTUAL GRANULARITY for understanding student learning.

        Scale interpretation for this item:
        - 1 = Option #1 is much more appropriate
        - 3 = No preference / roughly equal / insufficient evidence
        - 5 = Option #2 is much more appropriate

        {LIKERT_1_5_RUBRIC_COMMON}

        Evaluation criteria:
        - Match to instructional granularity in excerpts (learning objectives, assessment emphasis, module structure).
        - Prefer nodes that correspond to assessable competencies/misconceptions rather than incidental details.
        - Penalize overly generic nodes unless supported, and overly atomic nodes unless explicitly emphasized.

        Inputs:
        { _format_graph_option(option1) }

        { _format_graph_option(option2) }

        Instructor-material excerpts (evidence base):
        { _format_excerpts(excerpts) }

        { _json_output_contract() }
    """).strip()


# -----------------------------
# Metric (1) — Likert 1–5 prompts (overall graph)
# -----------------------------

def prompt_overall_dependent_concepts_helpfulness(graph_g: GraphOption, focus_concept: str, excerpts: List[Excerpt]) -> str:
    return textwrap.dedent(f"""
        You are evaluating an instructor-material-derived knowledge graph (or a large subgraph) G.

        Task:
        Rate how helpful G is for QUICKLY identifying dependent concepts when focusing on:
        Focus concept = "{focus_concept}"

        Interpretation:
        - Dependent concepts are those that require the focus concept as a prerequisite (i.e., concepts downstream of it via depends_on),
          and/or tightly connected learning dependencies supported by excerpts.

        Scale interpretation:
        1 = Not Helpful at All
        3 = Moderately Helpful / mixed / insufficient evidence
        5 = Extremely Helpful

        {LIKERT_1_5_RUBRIC_COMMON}

        Evaluation criteria:
        - Does G contain clear depends_on edges involving the focus concept (directly or via short paths)?
        - Are those dependencies plausible per excerpts (course sequencing, stated prerequisites)?
        - Is the neighborhood around the focus concept coherent (not cluttered by irrelevant nodes)?
        - If the focus concept is missing or disconnected, score low.

        Graph G:
        { _format_graph_option(graph_g) }

        Instructor-material excerpts (evidence base):
        { _format_excerpts(excerpts) }

        { _json_output_contract() }
    """).strip()


def prompt_overall_explain_structure_to_instructor(graph_g: GraphOption, excerpts: List[Excerpt]) -> str:
    return textwrap.dedent(f"""
        You are evaluating an instructor-material-derived knowledge graph (or a large subgraph) G.

        Task:
        Rate how useful G would be for explaining the conceptual structure of the course to another instructor or TA.

        Scale interpretation:
        1 = Not Useful at All
        3 = Somewhat Useful / mixed / insufficient evidence
        5 = Extremely Useful

        {LIKERT_1_5_RUBRIC_COMMON}

        Evaluation criteria:
        - Concept coverage: does G reflect the major themes and modules indicated by excerpts?
        - Organizational clarity: does depends_on and part_of convey a usable structure?
        - Terminology alignment: are labels consistent with course language?
        - Noise level: penalize irrelevant nodes/edges that would confuse a new TA/instructor.

        Graph G:
        { _format_graph_option(graph_g) }

        Instructor-material excerpts (evidence base):
        { _format_excerpts(excerpts) }

        { _json_output_contract() }
    """).strip()


def prompt_overall_explain_next_to_student(graph_g: GraphOption, student_goal: str, excerpts: List[Excerpt]) -> str:
    return textwrap.dedent(f"""
        You are evaluating an instructor-material-derived knowledge graph (or a large subgraph) G.

        Task:
        Rate how useful G would be for explaining to a student which concepts they should review next and WHY,
        given the student context/goal:
        Student context/goal = "{student_goal}"

        Scale interpretation:
        1 = Not Useful at All
        3 = Somewhat Useful / mixed / insufficient evidence
        5 = Extremely Useful

        {LIKERT_1_5_RUBRIC_COMMON}

        Evaluation criteria:
        - Are prerequisite (depends_on) paths clear enough to justify "review next" recommendations?
        - Are part_of relations helpful for narrowing into subskills?
        - Does G avoid misleading recommendations due to incorrect direction/type?
        - Are concepts expressed at a level students could act on?

        Graph G:
        { _format_graph_option(graph_g) }

        Instructor-material excerpts (evidence base):
        { _format_excerpts(excerpts) }

        { _json_output_contract() }
    """).strip()


def prompt_overall_correctness_interpretability_rating(graph_g: GraphOption, excerpts: List[Excerpt]) -> str:
    return textwrap.dedent(f"""
        You are evaluating an instructor-material-derived knowledge graph (or a large subgraph) G.

        Task:
        Provide an overall rating of the correctness AND interpretability of G as a tool for understanding student learning.

        Scale interpretation:
        1 = Very Poor
        3 = Adequate / mixed / insufficient evidence
        5 = Excellent

        {LIKERT_1_5_RUBRIC_COMMON}

        Evaluation criteria (roughly equal weight):
        Correctness:
        - Nodes correspond to real course concepts supported by excerpts.
        - Edge types/directions match prerequisite/subtopic relationships in excerpts.
        - Minimal contradictions, spurious links, or missing core areas.

        Interpretability:
        - Graph is coherent: clusters/modules make sense; limited noise.
        - Granularity is appropriate for learning diagnosis.
        - Relationships are easy to use for reasoning about what to learn next.

        Graph G:
        { _format_graph_option(graph_g) }

        Instructor-material excerpts (evidence base):
        { _format_excerpts(excerpts) }

        { _json_output_contract() }
    """).strip()


# -----------------------------
# Prompt registry (all Metric 1 prompts)
# -----------------------------

METRIC1_PROMPTS: Dict[str, object] = {
    # Ordinal 0–2 (node/triplet)
    "node_significance": prompt_m1_concept_node_validity_ordinal,
    "triplet_accuracy": prompt_m1_concept_triplet_accuracy_ordinal,

    # Likert 1–5 (pairwise)
    "m1_compare_major_concepts_likert_1_5": prompt_compare_major_concepts,
    "m1_compare_confidence_completeness_relevance_likert_1_5": prompt_compare_confidence_completeness_relevance,
    "m1_compare_relationship_accuracy_likert_1_5": prompt_compare_relationship_accuracy,
    "m1_compare_granularity_likert_1_5": prompt_compare_granularity,

    # Likert 1–5 (overall graph)
    "m1_overall_dependent_concepts_helpfulness_likert_1_5": prompt_overall_dependent_concepts_helpfulness,
    "m1_overall_explain_structure_to_instructor_likert_1_5": prompt_overall_explain_structure_to_instructor,
    "m1_overall_explain_next_to_student_likert_1_5": prompt_overall_explain_next_to_student,
    "m1_overall_correctness_interpretability_likert_1_5": prompt_overall_correctness_interpretability_rating,
}