# Neural Caissa

This is yet another attempt to build a neural network improved chess bot, which I named `neural caissa`. For those who don't know, Caissa is the [goddess of Chess ](https://en.wikipedia.org/wiki/Ca%C3%AFssa). These neural chess explorations were **heavily** inspired by George Hotz's [Twichchess](https://github.com/geohot/twitchchess).

How to play
-----

First you need to setup your virtual environment (maybe following [my recommended strategy](https://glhuilli.github.io/virtual-environments.html)).

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

TBA.


List of TODOs 
----
1. Speed up and/or distribute the serialization strategy for the Tensor representation 12x8x8 of the Board's state. (currently it's about ~ 6it/s), which it's slow if you consider that the training data is about 10M of games (~38M iterations if you consider that in average there's 38 moves per game [[source](https://chess.stackexchange.com/questions/2506/what-is-the-average-length-of-a-game-of-chess#:~:text=The%20average%20number%20of%20moves%20per%20game%20is%20around%2038.)]). This would take ~73 days to process.   
2. Improve space search strategy when running the Minimax algorithm (note that it's already including the alpha-beta pruning strategy) (@geohot in Twitchchess mentions using [Beam search](https://medium.com/@dhartidhami/beam-search-in-seq2seq-model-7606d55b21a5) for this).
3. Implement a `MuZero` style strategy to learn how to play chess from the game itself, instead of relying on the minimax backtracking algorithm to find the best next move (maybe something like [this](https://medium.com/applied-data-science/how-to-build-your-own-muzero-in-python-f77d5718061a))   
4. Improve the `ChessConvNet` structure so it's better suited for Chess. Currently is very simple stack of ConvNets, and most likely not using the right representation. I think that it can be particularly improved by using a different loss function (I personally like the loss function proposed by @erikbern in his [deep-pink implementation](https://github.com/erikbern/deep-pink)).  
5. Improve the serialization strategy as the current one is extremely simple.
6. Update the training script so it can be run in [Google Collab](https://pytorch.org/tutorials/beginner/colab.html) (or somewhere that it can use the GPU for training). 
