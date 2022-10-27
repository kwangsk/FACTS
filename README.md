# FACTS-model
This is a public repository of Python code to implement the Feedback Aware Control of Tasks in Speech (FACTS) model of speech motor control.

---Python Virtual Environment---

In Anaconda Prompt, run:
	conda env create -f environment.yml

Then run:
	conda activate facts_env
	
--- FACTS ---

Usage:
	python facts.py [config filepath .ini] [Gestural Score filepath]
	
For example:
	python facts.py lwpr_configs.ini GesturalScores/TVtest.G