# TRM reproduction

I went with the Sudoku-Hard, because it requires little compute while providing an interesting insight into the MLP-mixer vs self-attention mechanisms.
I ran all of it on an RTX A6000. 

Attention took 17h 47m, MLP 22h 39m. Both converged nicely without any hickups (which is quite rare out of the box for a research publication - solid job on Alexia Jolicoeur-Martineau's part).

Tracked all of it on W&B, I'll go over the most relevant params and explain them briefly below:
1. lm_loss (Language Modeling / Prediction Loss):
    Standard Cross-Entropy loss.
    It measures how good the model is at filling in the missing Sudoku cells.
    Intuitively: "Did it guess the number in cell (3,4) correctly?"

![lm_loss.png](lm_loss.png)

2. q_halt_loss (Halting Loss):
   Context: The TRM uses a mechanism inspired by Adaptive Computation Time (ACT). It tries to learn when it has thought enough.
   In her paper "Less is More" (TRM), Alexia simplifies the complex Q-learning approach used in previous models (HRM) to a simpler binary classifier.
   The q_halt_loss trains a small "head" on the network that outputs a probability: "Is my current answer correct yet?"
   It is trained using Binary Cross Entropy (BCE) against the ground truth (i.e., if the puzzle is currently solved, the target is 1; otherwise, 0).
   The name q_ is a legacy artifact from the HRM code (which used Q-learning) -> in TRM, it's effectively a "Halting Classifier Loss."

![q_halt_loss.png](q_halt_loss.png)
