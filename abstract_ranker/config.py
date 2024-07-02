from pathlib import Path


CACHE_DIR = Path("./.abstract_cache")

# Raw prompt for the LLM
abstract_ranking_prompt = """Help me judge the following conference presentation as interesting or
not by summarizing the abstract and ranking it according to topics I'm interested in or not.
"""

interested_topics = [
    "Hidden Sector Physics",
    "Long Lived Particles (Exotics or RPV SUSY)",
    "Analysis techniques and methods and frameworks, particularly those based around python or "
    "ROOT's DataFrame (RDF)",
    "Machine Learning and AI for particle physics",
    "The ServiceX tool",
    "Distributed computing for analysis (e.g. Dask, Spark, etc)",
    "Data Preservation and FAIR principles",
    "Differentiable Programming",
]

not_interested_topics = [
    "Quantum Computing",
    "Lattice Gauge Theory",
    "Neutrino Physics",
]
