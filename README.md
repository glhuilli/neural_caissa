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

