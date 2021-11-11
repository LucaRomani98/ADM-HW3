def AlgorithmicQuestion(Input):
    Inclusive, Exclusive = 0, 0
    for i in Input:
        Temp = Inclusive
        Inclusive = max(Inclusive, Exclusive + i)
        Exclusive = Temp
    return max(Inclusive, Exclusive)
