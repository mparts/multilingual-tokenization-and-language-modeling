# SAVED RUNS LIST
## Epochs = 10
### 1)  1_10E_Char_Char2k-10k_Byte2k-10k
- Goal:
    - First complete train. Get a grasp of where the models generally stand and what they represent. 
- Models Used:
    1) Character-Level
    2) CharLevelBPE_2000
    3) CharLevelBPE_10000
    4) ByteLevelBPE_2000
    5) ByteLevelBPE_10000
### 2) 2_10E_CharBPE-10k-11k-12k-15k-20k
- Goal:
    - Test out how Character-Level BPE performs with bigger vocabulary. Generalizing the results, it looks roughly something like this: __10k < ... < 20k__
- Models Used:
    1) CharLevelBPE_10000
    2) CharLevelBPE_11000
    3) CharLevelBPE_12000
    4) CharLevelBPE_15000
    5) CharLevelBPE_20000
### 3) 3_10E_ByteBPE-2k-4k-6k-8k-10k
- Goal:
    - Test out how Byte-Level BPE performs under different vocabularies between 2k and 10k.. I wanted to see if something like this was the case: 2k< 4-6-8 > 10k. Turns out, it roughly looks like this: __2k < ... < 10k__
- Models Used:
    1) ByteLevelBPE_2000
    2) ByteLevelBPE_4000
    3) ByteLevelBPE_6000
    4) ByteLevelBPE_8000
    5) ByteLevelBPE_10000
### 4) 6_10E_ByteBPE-10k-12k-14k-16k-18k-20k
- Goal:
    - Continue the tests on the Byte-Level BPE, now on higher values than 10k. Wanted to see if it can further improve under higher vocabulary or not. Turned out quite stable. Meaning that we increase computation cost, with minimal improvement. So basically no meaningfull gain, more parameters.
- Models Used:
    1) ByteLevelBPE_10000
    2) ByteLevelBPE_12000
    3) ByteLevelBPE_14000
    4) ByteLevelBPE_16000
    5) ByteLevelBPE_18000
    6) ByteLevelBPE_20000
## Epochs = 20
### 1) 4_20E_Char_Byte2k-10k-20k_Char20k
- Goal:
    - Big 20 epochs train. Chose five 'representative' models, out of my list and trained for 20 epochs to find out if the model can learn more if we expose it to more data, or if/when it starts to overfit.
- Models Used:
    1) Character-Level
    2) ByteLevelBPE_2000
    3) ByteLevelBPE_10000
    4) ByteLevelBPE_20000
    5) CharLevelBPE_20000
### 2)  5_20E_TEST
- Goal:
    - First run on the __TEST__ set. Exact same conditions as above, but models evaluated on test instead of valid. (Will probably use these results for my main conclusions)
- Models Used:
    1) Character-Level
    2) ByteLevelBPE_2000
    3) ByteLevelBPE_10000 
    4) ByteLevelBPE_20000
    5) CharLevelBPE_20000

## Seed != 0
### 1) 7_Seed1_Char_Byte10k_Char20k
- Goal:
    - Run 10 epochs, on 3 'representative' models, using a different seed, to help approximate seed noise, compared to seed 0. Did 2 of those runs (see bellow). Results can be quite rushed, since its only 3 models out of many, and only 3 total seeds, but results generally seem super stable, so I would naively suppose that seed noise is really really small.
- Models Used:
    1) Character-Level
    2) ByteLevelBPE_10000 
    3) CharLevelBPE_20000
### 2) 8_Seed2_Char_Byte10k_Char20k
- Goal:
    - Run 10 epochs, on 3 'representative' models, using a different seed, to help approximate seed noise, compared to seed 0. Did 2 of those runs (see above). Results can be quite rushed, since its only 3 models out of many, and only 3 total seeds, but results generally seem super stable, so I would naively suppose that seed noise is really really small.
- Models Used:
    1) Character-Level
    2) ByteLevelBPE_10000 
    3) CharLevelBPE_20000