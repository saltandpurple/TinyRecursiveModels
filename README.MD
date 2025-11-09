# TRM in games
Attempting to train a TRM on various games, most likely from google-deepmind's [Game Arena](https://github.com/google-deepmind/game_arena).

## ToDos:
1. Decide on a game. It must be turn-based, with text-based input/output (or easily convertible to such).
2. Decide on environment/framework. Could be game-arena mentioned above, but I'm also considering Unity ML Agents or simply writing the game myself.
3. Implement reinforcement learning / gym-like system.
4. Write training setup for TRM
5. Package to image
6. Debug the whole shit
7. Train it!

## Alternative considerations:
1. Move more towards computer vision and/or SLAM. Visual seems nice for me, since I respond strongly to it. 
2. How about video analysis? 
3. "Strategic AI": natural language -> symbols -Y symbolic reasoning engine -> LLM -> natural language
4. Analyze movies and predict the plot. 
5. "Acoustic AI" -> generate what, exactly?

## Training runs Sudoku-Hard
1. Multi-layer perceptron
```
run_name="pretrain_mlp_t_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

2. Attention layer
```
run_name="pretrain_att_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```