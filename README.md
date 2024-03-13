# abstract-ranker
Code to play with conference abstract ranking and classification

## Description

This is a hack. Trying to avoid writing the talk that I really needed to do, I wondered how well GPT would be able to rank the abstracts of the [ACAT workshop](https://indico.cern.ch/event/1330797) I was attending at the time. So I coded up some preferences in terms of keywords and asked it to look at abstracts and what they were connected with and write out a one-line summary of the abstract, interesting keywords, and to have it guess if I might be interested.

It worked surprisingly well. Enough to make me wonder if I shouldn't be using something like this with the archive. And to wonder if it shouldn't be able to figure out what I'm interested in by me selecting (or not) various talks/abstracts, and building my "like" and "not like" list on its own. Of course, what I like and don't like is partly affected by the conference I'm attending.

## Other info

Using gpt 3.5 turbo cost pannies to do this, and 4 turbo cost about $1.60 US. I didn't get as far as comparing the results of the two to know if the 3.5 was good enough.