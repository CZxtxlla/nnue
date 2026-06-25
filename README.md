# Introduction

---

The goal of this project is to implement an nnue with sparse matrix computations capable of being trained on the gpu using cuda. To do this I am taking my previous smallGrad project and stripping out all the fluff so that this is purely a nnue implementation. I will need to implement sparse matrix computation forward and backprop, as well as a custom loss function that takes into account the static evaluation as well as the game result.

Once the nnue is implemented I will implement a quanitization export so that the nnue trained here may be used in any engine that has the proper inference setup.

I will also be implementing a data generation pipeline, where you can specify the location of the executable to an engine and it will generate training data through self play, maybe with some filtering for quality of the positions.