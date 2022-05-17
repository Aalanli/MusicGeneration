# MusicGeneration
Classical music generation experiments with transformers

Network variants:

1. alibi transformer as described by: arXiv:2108.12409 [cs.CL]
2. Roformer as described by: 	arXiv:2104.09864 [cs.CL]
3. Relative positional embedding transformer as described by: arXiv:2009.13658 [cs.CL]

Observations

Relative positional embedding transformer performs similarly as the alibi embedding transformer, but with higher memory complexity. While Rotary embedding transformer performs the worst, with half the accuracy with similar memory complexity as the alibi transformer.
