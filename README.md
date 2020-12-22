# Neural Caissa

This is yet another attempt to build a neural network improved chess bot, which I named `neural caissa`. For those who don't know, Caissa is the [goddess of Chess ](https://en.wikipedia.org/wiki/Ca%C3%AFssa). These neural chess explorations were **heavily** inspired by George Hotz's [Twichchess](https://github.com/geohot/twitchchess).

How to play
-----

First you need to setup your virtual environment (maybe following [this recommended strategy](https://glhuilli.github.io/virtual-environments.html)).

Once this is ready, you can install all dependencies by just running, 

```bash
python setup.py install
```

Once all dependencies are ready, you can launch the service that is integrated with the bots trained in this repository by running, 

```bash
python run.py
```

Finally, you can play chess against Neural Caissa in your web browser by going to the following link: 

```
http://127.0.0.1:9000/
```

You can follow the runtime logs from the service and the web app by tailing the `record.log` file:

```bash
tail -f record.log 
```

How does it work
-----

In a nutshell, the current version works as follows:

1. An initial board `state` is generated and `valuator` strategy is set. The `valuator` will be used to compute the score for potential next states for the board.
2. Using a minimax algorithm with limited depth and alpha-beta pruning, a set of potential next states are explored. The `valuator` is used to compute the score for each state.  
3. The next state that has higher score, is used as the next move. 

The current `valuators` available are two: a baseline and a neural `valuator`. The baseline `valuator` is fairly simple, but works surprisingly ok. The current formula implemented is the following: 

```
value = SUM(piece in white)
        - SUM(piece in black)
        + 0.1 * (White pieces mobility score)
        - 0.1 * (Black pieces mobility score)
```

The neural `valuator` is a scoring function trained using a deep learning. This function is built using a series of ConvNets, which are then mapped into a linear representation and evaluated using a `tanh` into [-1, +1], using Adam as optimizer and MSE as the loss function, mini-batches of size 256, and 100 epochs to train.  

To train this model, a large collection of games where the features are determined by a very simple and naive serialization strategy: each state is represented as a Tensor with binary variables (12 x (8 x 8) variables). For each piece k (k \in [1, 12]), there's a 1 if piece k in position i else 0 (i \in [1, 64]). The target label is either +1 or -1 depending on whether the player that is moving in given state won the game or not. 

The data used to train this model was downloaded from [caissabase](http://caissabase.co.uk/). 

A more detailed description of the model is summarized in the following table:

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1             [-1, 16, 8, 8]           1,744
            Conv2d-2             [-1, 16, 8, 8]           2,320
            Conv2d-3             [-1, 32, 3, 3]           4,640
            Conv2d-4             [-1, 32, 3, 3]           9,248
            Conv2d-5             [-1, 32, 3, 3]           9,248
            Conv2d-6             [-1, 64, 1, 1]          18,496
            Conv2d-7             [-1, 64, 2, 2]          16,448
            Conv2d-8             [-1, 64, 3, 3]          16,448
            Conv2d-9            [-1, 128, 1, 1]          32,896
           Conv2d-10            [-1, 128, 1, 1]          16,512
           Conv2d-11            [-1, 128, 1, 1]          16,512
           Conv2d-12            [-1, 128, 1, 1]          16,512
           Linear-13                    [-1, 1]             129
================================================================
Total params: 161,153
Trainable params: 161,153
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.03
Params size (MB): 0.61
Estimated Total Size (MB): 0.65
----------------------------------------------------------------
```


List of TODOs 
----
1. Speed up and/or distribute the serialization strategy for the Tensor representation 12x8x8 of the Board's state. (currently it's about ~ 6it/s), which it's slow if you consider that the training data is about 10M of games (~38M iterations if you consider that in average there's 38 moves per game [[source](https://chess.stackexchange.com/questions/2506/what-is-the-average-length-of-a-game-of-chess#:~:text=The%20average%20number%20of%20moves%20per%20game%20is%20around%2038.)]). This would take ~73 days to process.   
2. Improve space search strategy when running the Minimax algorithm (note that it's already including the alpha-beta pruning strategy) (@geohot in Twitchchess mentions using [Beam search](https://medium.com/@dhartidhami/beam-search-in-seq2seq-model-7606d55b21a5) for this).
3. Implement a `MuZero` style strategy to learn how to play chess from the game itself, instead of relying on the minimax backtracking algorithm to find the best next move (maybe something like [this](https://medium.com/applied-data-science/how-to-build-your-own-muzero-in-python-f77d5718061a))   
4. Improve the `ChessConvNet` structure so it's better suited for Chess. Currently is very simple stack of ConvNets, and most likely not using the right representation. I think that it can be particularly improved by using a different loss function (I personally like the loss function proposed by @erikbern in his [deep-pink implementation](https://github.com/erikbern/deep-pink)).  
5. Improve the serialization strategy as the current one is extremely simple.
6. ~~Update the training script so it can be run in [Google Collab](https://pytorch.org/tutorials/beginner/colab.html) (or somewhere that it can use the GPU for training)~~. 
7. Improve minimax algorithm moving from depth-limited search to probability-limited search (similar to what was done in [Giraffe](https://arxiv.org/pdf/1509.01549.pdf)). Note that in the Giraffe paper there's a pretty interesting serialization strategy that accounts for ~300 features for a given board state. 
8. Maybe some simple Q-learning version might also be interesting to explore (Apparently in [this Kaggle](https://www.kaggle.com/arjanso/reinforcement-learning-chess-3-q-networks#Reinforcement-Learning-Chess) set of scripts there's something that coudl be useful). 
9. Consider using the Piece Square Tables used in [Sunfish](https://github.com/thomasahle/sunfish) (described also in this [blog post](https://dev.to/zeyu2001/build-a-simple-chess-ai-in-javascript-18eg)).
