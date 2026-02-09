
from typing import Optional, List, Dict
from typing_extensions import Literal
from pydantic import BaseModel, validator


class ConceptExtractionOutput(BaseModel):
    concepts: list[str]

class ChunkClassification(BaseModel):
    "YES or NO if a chunk contains important concepts."
    relevance: Literal["YES", "NO"]

class RoleTaggerOutput(BaseModel):
    "Role for concept in a chunk + a short evidence snippet."
    role: Literal["Example", "Definition", "Assumption", "NA"]
    snippet: str

class RelationJudgmentOutput(BaseModel):
    A: str
    B: str
    relation: Optional[Literal["depends_on", "part_of"]]  # null => no edge
    justification: str
    evidence: List[Dict]  # keep flexible; you can tighten later


ROLE_CLASSIFICATION_PROMPT = """
You will be given a text chunk from university lecture notes or slides and a concept from the course.
Classify the role the concept plays in this chunk into one of four categories:

1**Definition**: The concept is being defined, explained, or introduced. The text describes what the concept is, its properties, or how it works.
Simple example: "Binary search is an algorithm that finds an element in a sorted array by repeatedly dividing the search interval in half."
Complex example: 
- Concept: "Big O notation"
- Text: "When analyzing algorithm efficiency, we need a formal way to describe performance. Big O notation provides an upper bound on the growth rate of an algorithm's time complexity, expressing how runtime scales with input size n."
- Classification: Definition (the concept itself is being explained, even when embedded in broader context)

2**Example**: The concept is being demonstrated or illustrated through a concrete example, walkthrough, or application. The text shows the concept in action.
Simple example: "Let's apply binary search to find 7 in [1,3,5,7,9,11]: First, check the middle element 5..."
Complex example:
- Concept: "Recursion"
- Text: "To understand how the call stack works, consider computing factorial(3). The function calls factorial(2), which calls factorial(1), which calls factorial(0) returning 1. Then factorial(1) returns 1×1=1, factorial(2) returns 2×1=2, and finally factorial(3) returns 3×2=6."
- Classification: Example (shows recursion through a concrete walkthrough, even if framed as explanation)

3**Assumption**: The concept is being used as prior knowledge, a prerequisite, or a foundation for explaining something else. The text assumes familiarity with the concept to build further understanding.
Simple example: "Using binary search, we can now efficiently implement the dictionary lookup feature..."
Complex example:
- Concept: "Hash functions"
- Text: "Hash tables achieve O(1) average-case lookup because hash functions distribute keys uniformly across buckets. However, we must handle collisions when multiple keys map to the same index."
- Classification: Assumption (hash functions are used as known background to explain hash table behavior)

4** NA **: The concept does not fall under any of the above mentioned roles: (Definition, Example, Assumption).


**Key distinction**: If the concept is being taught → Definition. If it's being shown in action → Example. If it's being used to explain something else → Assumption. If none of the previously mentioned then -> NA


Also return an evidence snippet:
- Must be an exact substring copied from the chunk (10–30 words)
- Should best support your classification
- Keep it concise and relevant

Return strict JSON format only:
{ "role": "Definition" | "Example" | "Assumption" | "NA", "snippet": "..." }
""".strip()

CHUNK_CLASSIFICATION_PROMPT = """
You are an academic course content classifier.

You will recieve a chunk extraced from a course material, you need to classify whether this text chunk is
ACADEMICALLY RELEVANT (contains course content, learning objectives, or topics)
or ADMINISTRATIVE/IRRELEVANT (contains logistics like office hours, grading policy, contact info, artifacts, unnecessary details that are not concepts).

Rules:
- YES → if it discusses course concepts, learning objectives, or knowledge areas.
- NO  → if it includes dates, locations, instructor info, policies, Zoom links, or grading tables, artifacts, or unnecessary details that are not concepts.

Return strict JSON:
{ "relevance": "YES" | "NO" }
""".strip()


CONCEPT_EXTRACTION_PROMPT = """
You are an instructor that extracts learning concepts from text. 

- Concept: a core idea or topic about the subject matter. Only extract meaningful course concepts.
DO NOT extract example values, variable names, numbers, formulas, or code elements.
Ignore content inside examples, formulas.


Return strict JSON:
{ "concepts": ["..."] }

Rules:
- 1–5 words per concept
- No code tokens, variable names, numbers, example values
- Deduplicate
""".strip()


RELATION_JUDGMENT_PROMPT_TEMPLATE = """
You are an expert course instructor building a concept hierarchy for a university course.

TASK:
- Use ONLY these relations: ["depends_on","part_of"].
- For this ordered pair (A,B), choose the MOST DOMINANT relation if any exists.
- **CRITICAL**: The relation direction is ALWAYS from A to B (A → B). Do NOT reverse the direction.
- Dominance rules: prefer the single relation that best supports learning order and course understanding.
- Be minimal: avoid drawing edges just because they are possible. Only include edges that are clearly supported.
- If two relations both plausibly hold, output the dominant one only; output the other ONLY if strongly justified (lower confidence).
- You can skip making a relationship if there is no clear relation.
CAUTION: Never make a relationship between concepts that have no clear connection.

Definitions:
- Concept: a distinct course idea/skill/topic.
- Role: how a concept appears in a passage.
  - Definition = the passage defines/explains/introduces the concept.
  - Example = the passage demonstrates the concept via a concrete instance/walkthrough.
  - Assumption = the passage uses the concept as prior knowledge without teaching it.

RELATION DEFINITIONS & EXAMPLES (direction is ALWAYS A → B):

1. "depends_on": A depends_on B means A requires B as a prerequisite (B must be learned BEFORE A).
   **Direction**: A → B means "A requires B first"
   Examples:
   - "gradient descent" depends_on "derivatives" 
     → gradient descent (A) requires derivatives (B) as prerequisite
   - "backpropagation" depends_on "chain rule" 
     → backpropagation (A) requires chain rule (B) as prerequisite
   - "ANOVA" depends_on "variance" 
     → ANOVA (A) requires variance (B) as prerequisite
   
   **WRONG examples (these would be backwards)**:
   - DO NOT say "derivatives" depends_on "gradient descent" (this is reversed!)
   - DO NOT say B depends_on A when A actually needs B first

2. "part_of": A part_of B means A is a component/subtype/member of the broader concept B.
   **Direction**: A → B means "A is part of B" (A is the specific, B is the general)
   Examples:
   - "convolutional layer" part_of "neural networks" 
     → convolutional layer (A) is a component of neural networks (B)
   - "t-test" part_of "hypothesis testing" 
     → t-test (A) is a specific type within hypothesis testing (B)
   - "supervised learning" part_of "machine learning" 
     → supervised learning (A) is a subcategory of machine learning (B)
   
   **WRONG examples (these would be backwards)**:
   - DO NOT say "neural networks" part_of "convolutional layer" (this is reversed!)
   - DO NOT say B part_of A when A is actually the specific case of B

PAIR:
A = "{A}"
B = "{B}"

ROLES:
{ROLE_BLOCK}

TEMPORAL / STATS:
{TEMPORAL_BLOCK}

EVIDENCE_MODE:
{MODE_BLOCK}

RULES FOR USING THE EVIDENCE TEXT: - Base your decision ONLY on the text shown in EVIDENCE below. - The EVIDENCE text is the supporting passages for this (A,B) pair. - If the text does not clearly support a relation, output null. - Your returned evidence[].quote MUST be an exact substring copied from the provided text.

EVIDENCE (use ONLY what is shown below; do not assume anything unstated):
{EVIDENCE_BLOCK}

Return strict JSON ONLY (no markdown, no extra text):
{{
  "A": "{A}",
  "B": "{B}",
  "relation": "depends_on" | "part_of" | null,
  "confidence": 0.0,
  "justification": "1-3 sentences grounded in the evidence above",
  "evidence": [
    {{"type": "chunk" | "cluster", "chunk_id": "...", "lecture_id": "...", "page_numbers": [1,2], "quote": "..."}}
  ]
}}
""".strip()