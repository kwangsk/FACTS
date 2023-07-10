# Feedback Aware Control of Tasks in Speech (FACTS)
This is a public repository of Python code to implement the Feedback Aware Control of Tasks in Speech (FACTS) model of speech motor control.

This code can run the Python-implemented version of the previous FACTS (Parrell et al., 2019) as well as the recently developed version (see Design C in Kim et al., in review).

For detailed descriptions of design features and modules, please refer to Kim et al. (in review). 

## Pre-installation requirements
You will need to have Python installed. We have used 3.6 but more recent version should work too. FACTS can run on Windows, Mac OS, and Ubuntu.

## Installation steps
1. [Python Virtual Environment](#python-virtual-environment)
2. [Support Files](#support-files)

## Python Virtual Environment

In Anaconda Prompt, run
```gitbash
	conda env create -f environment.yml
```
Then run:
```gitbash
	conda activate facts_env
```
	
## Support Files 
Please follow the steps at [the FACTS-support repo](https://github.com/kwangsk/FACTS-support)


## How to run

### If you are running simulations included in Kim et al. (in press)
Make sure the line 119 is commented
```bash
#main(sys.argv[1:])
```
and uncomment a desired simulation (lines 121- 167)
For example:
```bash
	main(['DesignC_AUKF.ini','GesturalScores/KimetalAdapt.G'])
```

and run:
```bash
	python facts.py
```

### If you are running your own configuration file (.ini) and gestural score (.G)

```bash
	python facts.py [config filepath .ini] [Gestural Score filepath]
```

For example:
```bash
	python facts.py lwpr_configs.ini GesturalScores/TVtest.G
```

### If you want to save simulations
Make sure line 103 says True:
```bash
save = True
```
and set the datafile_name to a desired name in line 106 such as:
```bash
datafile_name = HierAUKF
```