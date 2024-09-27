# abstract-ranker

Code to play with conference or paper abstract ranking and classification

## Description

This is a python command-line tool, `abstract_ranker` that will fetch contributions from a public indico event (e.g. a conference) or yesterday's submissions from arXiv. It will then rank their interest from low (1) to high (3) against a list of topics you are interested in.

The output is a csv file, which can easily be loaded into Excel or Google Sheets and viewed.

### History

This is a hack. Trying to avoid writing the talk that I really needed to do, I wondered how well GPT would be able to rank the abstracts of the [ACAT workshop](https://indico.cern.ch/event/1330797) I was attending at the time. So I coded up some preferences in terms of keywords and asked it to look at abstracts and what they were connected with and write out a one-line summary of the abstract, interesting keywords, and to have it guess if I might be interested.

It worked surprisingly well. Enough to make me wonder if I shouldn't be using something like this with the archive. And to wonder if it shouldn't be able to figure out what I'm interested in by me selecting (or not) various talks/abstracts, and building my "like" and "not like" list on its own. Of course, what I like and don't like is partly affected by the conference I'm attending.

## Other info

Using gpt 4o-mini cost pennies to do this, and 4o cost about $0.70 US. I didn't get as far as comparing the results of the two to know if the 3.5 was good enough.

### Adding a new model

The file `llm_utils.py` contains a list of all models and how they are called. If you are adding a model from OpenAI or from hugging-face, it could be all you need to do is add a new entry to the dictionary in `llm_utils.py`. Worst case you will need to add a new query method - but you can use the hugging face and OpenAI ones as models for how to do this.

## Trying it out

The following is a way to use `phi-3` mini as an example:

```bash
 abstract_ranker --model phi3-mini -v rank_indico https://indico.cern.ch/event/1330797
```

You will need to create a `.openai_key` file in the directory you run this from. It should contains your OpenAI API key.

* Make sure only you can read it!
* THe file should contain the key and nothing else.

### Ranking indico abstracts

Any public indico event can have its contributions ranked by just giving the event URL. And output `csv` file will be made with the ranking in it.

```bash
 abstract_ranker --model phi3-mini -v rank_indico https://indico.cern.ch/event/1330797
```

### Ranking yesterday's arXiv upload

List the archive topics you are interested in and they will be ranked in a `arxiv-<topic>-<date>.csv` file.

```bash
 abstract_ranker --model GTP4o-mini -v rank_arxiv hep-ex
```

### Installing pytorch with CUDA

I had a lot of trouble here - so keeping a log:

1. Use `nvcc --version` to determine what `cuda` version you have on your local machine.
1. Use [these instructions](https://pytorch.org/get-started/locally/) to install pytorch. I did it in a conda environment, but a `venv` that uses `pip` should be fine too.
1. Use the usual `pip install -e .[test,ml]`
    * Note that this needs to be run on a Linux with a decent size GPU.
    * phi3-mini will work on a smaller GPU (like a 2080 TI).
    * phi3-small needs a modern GPU to run (like an A100 - not a V100!).
    * phi3-small requires something larger.

Notes:

* On UChicago use a 2080. The A100 MIG's do not have enough memory even for a mini.
