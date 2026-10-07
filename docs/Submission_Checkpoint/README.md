## Note to Teachers regarding checkpoint

I did really bad time management the past days, that's why I couldn't and didn't make the checkpoint. Regardless, currently the assingment progresses smoothly.

Regarding the code, asside from what I am uploading, I am almost done with the transformer model, but there are missing parts still, and what is completed needs cleaning up. So although really close to completing, it was in no state of submiting.

Regarding docs, asside from what I am uploading there are a few scattered thoughts I've put to notes, but too scattered and informal to submit.. they are there mostly for me to not forget them when the time comes to gather them in the analysis. Even those that I am currentlyn uploading aren't really that formal.. I am sorry for this, in the final submission, as always, I am going to go through everything and refine my wording.

### answers to questions:

#### the BPE vocabulary sizes you chose and why;

I chose 2k and 10k, which I think that they are far from ideal, since because of chinese, our base vocabulary (meaning the standard character-level) is arround 9k. So when I try to constrict to e.g. 2k, BPE doesn't even have enough room to learn the standard vocabulary, let alone merge. 10k merges some but still not ideal. Instead of playing arround with sizes I also implemented bytelevelBPE, which seems to be working alright for both 2k and 10k. I will have more insight during the evaluation of my transformer. Regard the above thoughts, again as 'hypotheses'. If there is time later down the road I will play arround with sizes.

#### your initial hypotheses about how the three tokenizers will behave;

notes.md contains my initial hypotheses about the tokenizers (again, sorry for the informality of the document)

#### preliminary tokenizer statistics for the three languages;

inside tokenizer_stats.json

#### one English, one Turkish, and one Chinese sentence tokenized by all three tokenizers;

contained inside tokenized_sentences.txt, produced by examine_tokenizers.py (examine_tokenizers.py currently only prints in console, I manually coppied them into the txt just for the submission.. I will later implement automatic saving into docs)

#### one question that you want your final experiments to help answer.

I am heavily interested to see how well the bytelevelbpe does compared to the rest tokenizers. And in what ways I coudle further improve.

#### one result or observation that you find interesting so far;

I haven't spent TOO MUCH time on it yet, but my first thoughts on the regular 10k BPE is that it seems to be doing way better than what I thought it will, in the begining. Maybe I am wrong and it isn't actually doing that well, because I might be reading my stats wrong.. Or maybe I shouldn't had underestimated it. But given that our character-level vocabulary is arround 9k, which means that the BPE only learns about half a thousand pairs/merges, based on the statisticts, it seems to be holding it's own nicely!!
