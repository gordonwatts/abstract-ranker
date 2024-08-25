# abstract-ranker

Code to play with conference abstract ranking and classification

## Description

This is a hack. Trying to avoid writing the talk that I really needed to do, I wondered how well GPT would be able to rank the abstracts of the [ACAT workshop](https://indico.cern.ch/event/1330797) I was attending at the time. So I coded up some preferences in terms of keywords and asked it to look at abstracts and what they were connected with and write out a one-line summary of the abstract, interesting keywords, and to have it guess if I might be interested.

It worked surprisingly well. Enough to make me wonder if I shouldn't be using something like this with the archive. And to wonder if it shouldn't be able to figure out what I'm interested in by me selecting (or not) various talks/abstracts, and building my "like" and "not like" list on its own. Of course, what I like and don't like is partly affected by the conference I'm attending.

## Other info

Using gpt 3.5 turbo cost pennies to do this, and 4 turbo cost about $1.60 US. I didn't get as far as comparing the results of the two to know if the 3.5 was good enough. And the updated models are even cheaper. 4o is $0.70, and 4o-mini is just 2 cents.

### Adding a new model

The file `llm_utils.py` contains a list of all models and how they are called. If you are adding a model from OpenAI or from hugging-face, it could be all you need to do is add a new entry to the dictionary in `llm_utils.py`. Worst case you will need to add a new query method - but you can use the hugging face and OpenAI ones as models for how to do this.

## Tyring it out

The following is a way to use `phi-3` mini as an example:

```bash
 abstract_ranker rank https://indico.cern.ch/event/1330797/contributions --model phi3-mini -v
```

### Installing pytorch with cuda

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
