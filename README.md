# chess-vision

chessvision calculates the state of a real chessboard from a video stream.
Necessary if we are to unleash computers superhuman chess ability into the real world,
but also an excuse to train some computer vision models and collect my own dataset!

Limitations:

 - **Output is only concerned with the board state** (not position of pieces in world space)
 - **Only one chess board is used in training** (other boards can be used but model will need to be retrained)
 - **Camera is assumed to be mounted directly above the chess board**(small perspective shift is accounted for, but extreme angles where occlusion will be a problem is not)


[guild](https://guild.ai/) was used in this project for experiment tracking.  Highly recommend you go check it out if you haven't already.

## Exploring The Code

**Training**
1. Visit [train.py](chessvision/train.py) for a view of the training script
2. Visit [models.py](chessvision/train.py) to see some of the models that were tested

**Auto-Labelling**

3. Visit [game.py](chessvision/game.py) to see how one "unit" of data is stored
4. Visit [label.py](chessvision/label.py) for the actual auto-labelling functions using [python-chess](https://github.com/niklasf/python-chess)
5. Use [record.py](chessvision/record.py) and [replay.py](chessvision/replay.py) to record your own games

**Inference**

6. Explore [inference.py](chessvision/inference.py) to see how the pre-trained models are used at runtime with the concepts of `VisionState` and `BoardState`
