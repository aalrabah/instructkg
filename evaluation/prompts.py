from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional
import textwrap
from pydantic import BaseModel


# -----------------------------
# Data structures (inputs)
# -----------------------------

@dataclass(frozen=True)
class Excerpt:
    """
    An excerpt from instructor-provided material.
    excerpt_id: stable identifier (e.g., "lec03_p2_s5" or "hw1_solution_par3").
    text: excerpt content (pre-trim upstream if very large).
    """
    excerpt_id: str
    text: str


@dataclass(frozen=True)
class Edge:
    """
    A typed directed edge between two concept nodes.
    relation_type must be one of: "depends_on" | "part_of"
    """
    source: str
    relation_type: str  # "depends_on" | "part_of"
    target: str


@dataclass(frozen=True)
class GraphOption:
    """
    A candidate graph/subgraph option for comparison tasks, or a single graph G for overall tasks.
    """
    option_id: str  # e.g., "Option #1", "Option #2", or "Graph G"
    nodes: List[str]
    edges: List[Edge]


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
    Use the following 0–2 ordinal scale strictly (Concept Node validity/meaningfulness):

    0 = No, not at all (invalid OR not meaningful for this course based on excerpts)
    1 = Somewhat (valid but not very meaningful / only weakly supported / marginal)
    2 = Yes (both valid and meaningful; clearly supported by excerpts)

    Evidence policy:
    - Base your score ONLY on the provided excerpts and the node label.
    - If evidence is insufficient or ambiguous, score 1 and state what is missing.
    - Cite evidence by excerpt_id (avoid long quotations).
""").strip()


ORDINAL_0_2_TRIPLET_RUBRIC = textwrap.dedent("""
    Use the following 0–2 ordinal scale strictly (Concept Triplet / Edge accuracy):

    0 = No (A and B should not be directly linked)
    1 = Somewhat (A and B should be directly related, but the type and/or direction is different)
    2 = Yes (A and B are directly related according to the edge type and direction shown)

    Evidence policy:
    - Base your score ONLY on the provided excerpts and the proposed edge.
    - If evidence is insufficient or ambiguous, score 1 and state what is missing.
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
) -> str:
    """
    Metric (1) Concept Node question (0–2):
    "Is node A a valid, meaningful concept for your course...?"
    """
    return textwrap.dedent(f"""
        You are evaluating a single concept node extracted for a course knowledge graph.

        Task:
        Score whether the concept node is a valid, meaningful concept for the course,
        based strictly on the instructor-provided excerpts.

        Concept Node:
        - A = "{node_label}"

        {ORDINAL_0_2_NODE_RUBRIC}

        Instructor-material excerpts (evidence base):
        { _format_excerpts(excerpts) }

        { _json_output_contract() }
    """).strip()


def prompt_m1_concept_triplet_accuracy_ordinal(
    edge: Edge,
    excerpts: List[str],
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

        Concept Triplet:
        - A = "{edge.source}"
        - relation = "{edge.relation_type}"  (allowed: depends_on, part_of)
        - B = "{edge.target}"
        Interpreting relation types:
        - depends_on: A is a prerequisite of B (A should be learned before B)
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