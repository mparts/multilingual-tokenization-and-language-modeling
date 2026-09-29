### Predictions for Part 1
- Which language do you expect to require the most tokens?  
I expect Chinese to require the most tokens, since it seems to be the more diverse of the three. So BPE will probably have a harder time with Chinese in contrast to English or Turkish.


- How do you expect the larger BPE vocabulary to change the representation of each language?    
For all three languages, the larger the BPE gets, I expect the tokens per sentence to decrease. I suppose that Chinese should lead. Meaning that I suspect it will be the language benefiting more from increasing the size of the BPE.

- What kinds of units do you expect BPE to learn in English, Turkish, and Chinese?  
I am not sure about chinese, but I suppose the most frequent symbol combinations? Or maybe symbols that might represent single words. About english and turkish I suspect common suffixes and prefixes. Also probabbly combinations like 'and', 'the', etc..

- Which tokenizer do you expect to work best for language modeling?     
I suspect the middle ground, that being a moderately sized BPE. That is because compared to a standard character-level tokenizer I think that there are benefits in using BPE. Setting this aside and comparing a bigger BPE, with a smaller BPE, I feel like this depends highly on the target language. Supposedely, if my understanding of BPE is correct, a smaller BPE might favour one language (e.g. English) and a larger one another (e.g. Chinese).