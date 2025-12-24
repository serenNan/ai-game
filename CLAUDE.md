# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaZero-Gomoku is a Python implementation of the AlphaZero algorithm for Gomoku (Five in a Row). It uses deep reinforcement learning with Monte Carlo Tree Search (MCTS) to train an AI through self-play.

## Commands

### Play with GUI
```bash
python main.py
```
Launches Pygame GUI with Human vs AI modes. Uses pure NumPy inference with pre-trained model (`8_8_5.model`).

### Train from scratch
```bash
python trainer.py
```
Default uses Theano/Lasagne. To use a different framework, modify the import in `trainer.py`:
```python
# Choose one:
from model_theano import NeuralNetworkEvaluator  # Theano/Lasagne (default)
from model_torch import NeuralNetworkEvaluator   # PyTorch
from model_tf import NeuralNetworkEvaluator      # TensorFlow
from model_keras import NeuralNetworkEvaluator   # Keras
```

## Architecture

### Core Components

- `board.py`: Game engine - `GameState` (board state, 4-channel representation) and `GameController` (game flow, self-play)
- `neural_search.py`: MCTS with neural network guidance - `SearchNode`, `MonteCarloTreeSearch`, `TreeSearchAgent`
- `random_search.py`: Pure MCTS baseline (random rollouts) for evaluation
- `model_*.py`: Policy-value networks (Theano, PyTorch, TensorFlow, Keras implementations)
- `model_inference.py`: NumPy-only inference, loads Theano weights without requiring the framework
- `main.py`: Pygame GUI

### Training Pipeline (`trainer.py`)

```
Self-Play → Data Augmentation (8x via rotations/flips) → Policy Update → Evaluation → Checkpoint
```

Key parameters in `TrainingManager`:
- `boardCols`, `boardRows`, `winLength`: Board configuration (default 6x6, 4-in-row)
- `numSimulations=400`: MCTS simulations per move
- `miniBatchSize=512`, `replayBufferSize=10000`: Training buffer
- `klTarget=0.02`: KL-divergence target for early stopping
- `evaluationInterval=50`: Evaluation frequency

### Data Flow

1. `GameController.runSelfPlay()` generates (state, mcts_probs, winner) tuples
2. States augmented 8-fold via `augmentData()`
3. `NeuralNetworkEvaluator.trainOnBatch()` updates network
4. MCTS tree reused across moves within a game for efficiency

## Pre-trained Models

- `6_6_4.model`: 6x6 board, 4-in-a-row
- `8_8_5.model`: 8x8 board, 5-in-a-row

Models are in Theano/Lasagne pickle format. `model_inference.py` loads these for inference without Theano.

## Dependencies

**Playing only**: NumPy >= 1.11, Pygame

**Training**: Choose one deep learning framework (Theano+Lasagne, PyTorch, TensorFlow, or Keras)
