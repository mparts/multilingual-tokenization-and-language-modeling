Multilingual tokenization assignment data
========================================

Languages: English (en), Turkish (tr), Mandarin Chinese (zh)
Source: Leipzig Corpora Collection, Wikipedia sentence corpora, 2021, 100K

Splits
------
train/en.txt, train/tr.txt, train/zh.txt
valid/en.txt, valid/tr.txt, valid/zh.txt
test/en.txt,  test/tr.txt,  test/zh.txt

Tokenizer-training corpora
--------------------------
tokenizer/balanced.txt
    Combined training sentences with approximately equal underlying character counts per language.

tokenizer/imbalanced_10-1-1.txt
    Same training pool, but approximately EN:TR:ZH = 10:1:1 by Unicode character count.
    Intended for the optional imbalance experiment. Do not use it as LM evaluation data.

Important design choice
-----------------------
The languages are balanced by Unicode character count rather than sentence count. This is not a
claim that one Unicode character carries equal linguistic information across languages; it is simply
a transparent, tokenizer-independent unit for constructing the corpus. Students can
discuss the limitations of this choice.

Provenance and exact counts are in metadata.json.

License / attribution
---------------------
See Leipzig's current terms of use: https://wortschatz.uni-leipzig.de/en/usage
Please cite the Leipzig Corpora Collection / Goldhahn, Eckart & Quasthoff (LREC 2012).
