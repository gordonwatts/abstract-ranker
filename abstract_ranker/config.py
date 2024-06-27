from pathlib import Path


CACHE_DIR = Path("./.abstract_cache")

# Raw prompt for the LLM
abstract_ranking_prompt = """Help me judge the following conference presentation as interesting or
not. My interests are in the following areas:

    1. Hidden Sector Physics
    2. Long Lived Particles (Exotics or RPV SUSY)
    3. Analysis techniques and methods and frameworks, particularly those based around python or
       ROOT's DataFrame (RDF)
    4. Machine Learning and AI for particle physics
    5. The ServiceX tool
    6. Distributed computing for analysis (e.g. Dask, Spark, etc)
    7. Data Preservation and FAIR principles
    8. Differentiable Programming

I am *not interested* in:

    1. Quantum Computing
    2. Lattice Gauge Theory
    3. Neutrino Physics

"""
