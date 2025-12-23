# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaZero-Gomoku is a Python implementation of the AlphaZero algorithm for Gomoku (Five in a Row). It uses deep reinforcement learning with Monte Carlo Tree Search (MCTS) to train an AI through self-play.

## Commands

### Play against trained AI
```bash
python human_play.py
```
Uses pure NumPy inference with pre-trained model (`best_policy_8_8_5.model`).

### Train from scratch
```bash
python train.py
```
Default uses Theano/Lasagne. To use a different framework, modify the import in `train.py`:
```python
# Choose one:
from policy_value_net import PolicyValueNet              # Theano/Lasagne (default)
from policy_value_net_pytorch import PolicyValueNet      # PyTorch
from policy_value_net_tensorflow import PolicyValueNet   # TensorFlow
from policy_value_net_keras import PolicyValueNet        # Keras
```

## Architecture

### Core Components

**Game Engine** (`game.py`):
- `Board`: Game state management, legal moves, winner detection
- `Game`: Game flow orchestration, self-play data collection
- Board state represented as 4 channels: current pieces, opponent pieces, last move, color to play

**MCTS** (`mcts_alphaZero.py`):
- `TreeNode`: UCB1 selection with prior probabilities from policy network
- `MCTS`: Monte Carlo Tree Search with neural network guidance
- `MCTSPlayer`: AI player interface with optional Dirichlet noise for self-play exploration

**Pure MCTS** (`mcts_pure.py`):
- Baseline MCTS without neural network (uniform policy rollouts)
- Used for evaluation during training

**Policy-Value Networks**:
| File | Framework | Notes |
|------|-----------|-------|
| `policy_value_net.py` | Theano/Lasagne | Default, pre-trained models use this |
| `policy_value_net_pytorch.py` | PyTorch | GPU support |
| `policy_value_net_tensorflow.py` | TensorFlow | GPU support |
| `policy_value_net_keras.py` | Keras | - |
| `policy_value_net_numpy.py` | NumPy only | Inference only, loads Theano weights |

### Training Pipeline (`train.py`)

```
Self-Play → Data Augmentation (8x via rotations/flips) → Policy Update → Evaluation → Checkpoint
```

Key parameters in `TrainPipeline`:
- `board_width`, `board_height`, `n_in_row`: Board configuration
- `n_playout=400`: MCTS simulations per move
- `batch_size=512`, `buffer_size=10000`: Training buffer
- `kl_targ=0.02`: KL-divergence target for early stopping
- `check_freq=50`: Evaluation frequency

### Data Flow

1. `Game.start_self_play()` generates (state, mcts_probs, winner) tuples
2. States augmented 8-fold via `get_equi_data()`
3. `PolicyValueNet.train_step()` updates network
4. MCTS tree reused across moves within a game for efficiency

## Pre-trained Models

- `best_policy_6_6_4.model`: 6x6 board, 4-in-a-row
- `best_policy_8_8_5.model`: 8x8 board, 5-in-a-row (standard Gomoku)

Models are in Theano/Lasagne format. `policy_value_net_numpy.py` can load these for inference without Theano installed.

## Dependencies

**Playing only**: NumPy >= 1.11

**Training**: Choose one deep learning framework (Theano+Lasagne, PyTorch, TensorFlow, or Keras)
