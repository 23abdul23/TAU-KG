"""
Constants for the Biomedical QA System
Contains prompts, examples, and other static data
"""

# System prompt for generation - used to ensure strict output control
INFERENCE_SYSTEM_PROMPT = """
You are a specialized biomedical question answering system designed for scientists. Your responses must be:

1. PRECISE: Provide ONLY the exact information requested. If asked for gene names, list ONLY gene names.
2. CONCISE: No introductions, explanations, or summaries unless explicitly requested.
3. FACTUAL: Base all responses on scientific literature.
4. STRUCTURED: Present information in a clean, structured format (lists, tables) when appropriate.

For gene lists, protein, disease or pathway lists or other enumerations, use a simple bullet point format:
• GENE1
• GENE2

NEVER include any qualifying statements like "Based on the literature" or "Research suggests".
NEVER embellish answers with additional explanations.
DO use proper scientific notation, nomenclature, and symbols appropriate for biomedical context.
"""


"""
eval metric:
rouge : for the main summarization score
bleu score: 


Question: What genes are associated with Parkinson's disease?
Answer<biobert>:
Parkinson’s disease (PD) pathogenesis is underpinned by a constellation of genetic perturbations 
encompassing both Mendelian mutations and risk-conferring polymorphisms. Key loci include SNCA, 
whose gene product α-synuclein undergoes pathogenic fibrillization; LRRK2, encoding a multi-domain 
ROCO protein with GTPase and kinase activity implicated in vesicular trafficking and autophagy; PINK1
 and PRKN (PARK2), central to mitophagic clearance of depolarized mitochondria via the PINK1-Parkin 
 signaling axis; DJ-1 (PARK7), functioning as a redox-sensitive chaperone mitigating oxidative insult;
and GBA, where heterozygous mutations perturb lysosomal glucocerebrosidase function, amplifying 
α-synucleinopathy. Additionally, VPS35 mutations alter retromer complex-mediated endosomal sorting,
while CHCHD2, FBXO7, and ATP13A2 variants modulate mitochondrial cristae integrity, E3 ligase scaffolding,
and lysosomal cation homeostasis, respectively. Cumulatively, these genetic lesions converge on 
disrupted proteostasis, mitochondrial dyshomeostasis, neuroinflammation, and dopaminergic neuronal demise.


"""
# Few-shot examples to guide the model's output format
FEW_SHOT_EXAMPLES = """
Example 1:
Question: What genes are associated with Parkinson's disease?
Answer:
• SNCA
• LRRK2
• PARK7
• PINK1
• PRKN
• GBA
• VPS35




Example 2:
Question: What are the key proteins involved in the JAK-STAT signaling pathway?
Answer:
• JAK1
• JAK2
• JAK3
• TYK2
• STAT1
• STAT2
• STAT3
• STAT4
• STAT5A
• STAT5B
• STAT6

Example 3:
Question: What biological pathways are dysregulated in Alzheimer's disease?
Answer:
• Amyloid-beta metabolism
• Tau phosphorylation
• Neuroinflammation
• Mitochondrial dysfunction
• Calcium homeostasis
• Oxidative stress
• Autophagy-lysosomal system
• Insulin signaling
"""

# Gene patterns for regular expression matching
GENE_PATTERN = r'\b[A-Z][A-Z0-9]{1,}(?:-\d+)?\b'

# Protein patterns for regular expression matching
PROTEIN_PATTERN = r'\b[A-Z][a-z]*(?:-[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+){0,3}\s+(?:protein|receptor|kinase|phosphatase|enzyme|transporter|channel|factor)\b'

# Disease patterns for regular expression matching
DISEASE_PATTERN = r'\b(?:[A-Z][a-z]+\s+){1,4}(?:disease|disorder|syndrome|deficiency|cancer|tumor|carcinoma|leukemia|lymphoma)\b'

# Pathway patterns for regular expression matching
PATHWAY_PATTERN = r'\b(?:[A-Z][a-z]+\s+){0,3}(?:pathway|signaling|signalling|cascade|axis)\b'

# Common non-gene acronyms to filter out false positives
COMMON_NON_GENES = {
    "DNA", "RNA", "PCR", "THE", "AND", "NOT", "FOR", "THIS", "WITH", "FROM",
    "TYPE", "CELL", "CELLS", "FACTOR", "STUDY", "REVIEW", "HUMAN", "MOUSE",
    "RAT", "CASE", "REPORT", "ANALYSIS", "EFFECT", "EFFECTS", "ROLE",
    "ASSOCIATED", "ASSOCIATION", "INVOLVED", "PATHWAY", "RECEPTOR", "PROTEIN",
    "EXPRESSION", "LEVELS", "ACTIVITY", "REGULATION", "FUNCTION", "MUTATION",
    "MUTATIONS", "GENE", "GENES", "SNP", "SNPS", "MIRNA", "NCRNA", "LNCRA",
    "COVID", "SARS-COV-2", "AIDS", "HIV", "USA", "NIH", "FDA"
}

# Query type indicators for entity detection
QUERY_TYPE_INDICATORS = {
    "gene": ["gene", "genes", "mutation", "allele", "locus", "polymorphism", "variant"],
    "protein": ["protein", "receptor", "enzyme", "antibody", "kinase", "transporter"],
    "pathway": ["pathway", "signaling", "cascade", "metabolic", "process"],
    "disease": ["disease", "disorder", "syndrome", "condition", "pathology", "cancer"]
}

# Template prompts for different query types
QUERY_TYPE_INSTRUCTIONS = {
    "gene": """
    For gene-related questions:
    - List ONLY official gene symbols
    - Use HGNC nomenclature when applicable
    - Format gene symbols in all capitals (e.g., BRCA1, TP53)
    - Present as a bullet point list with no additional text
    - Do NOT include gene descriptions or functions unless explicitly asked
    """,

    "protein": """
    For protein-related questions:
    - List ONLY protein names
    - Use UniProt nomenclature when applicable
    - Present as a bullet point list with no additional text
    - Include protein complexes as separate entries when relevant
    - Do NOT include protein descriptions or functions unless explicitly asked
    """,

    "pathway": """
    For pathway-related questions:
    - List ONLY pathway names
    - Use standard pathway nomenclature (KEGG, Reactome)
    - Present as a bullet point list with no additional text
    - Do NOT include pathway descriptions or components unless explicitly asked
    """,

    "disease": """
    For disease-related questions:
    - List ONLY disease names
    - Use standard medical nomenclature
    - Present as a bullet point list with no additional text
    - Do NOT include disease descriptions or symptoms unless explicitly asked
    """,

    "general": """
    For scientific questions:
    - Provide ONLY the exact information requested
    - Present lists as bullet points
    - Do NOT include explanations or descriptions unless explicitly asked
    - Use standard scientific nomenclature
    """
}
