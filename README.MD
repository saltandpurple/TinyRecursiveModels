# TRM reproduction

I went with the Sudoku-Hard, because it requires little compute while providing an interesting insight into the MLP-mixer vs self-attention mechanisms.
I ran all of it on an RTX A6000. 

Attention took 17h 47m, MLP 22h 39m. Both converged nicely without any hickups (which is quite rare out of the box for a research publication - solid job on Alexia Jolicoeur-Martineau's part).

Tracked all of it on W&B. Since I had to look up what some of the params actually meant (the naming wasn't quite intuitive in all cases), I'll briefly explain those below:
1. **lm_loss (Language Modeling / Prediction Loss)**:
    Standard Cross-Entropy loss.
    It measures how good the model is at filling in the missing Sudoku cells.
    Intuitively: "Did it guess the number in cell (3,4) correctly?"

![lm_loss.png](lm_loss.png)

2. **q_halt_loss (Halting Loss)**:
   Context: The TRM uses a mechanism inspired by Adaptive Computation Time (ACT). It tries to learn when it has thought enough.
   In her paper "Less is More" (TRM), Alexia simplifies the complex Q-learning approach used in previous models (HRM) to a simpler binary classifier.
   The q_halt_loss trains a small "head" on the network that outputs a probability: "Is my current answer correct yet?"
   It is trained using Binary Cross Entropy (BCE) against the ground truth (i.e., if the puzzle is currently solved, the target is 1; otherwise, 0).
   The name q_ is a legacy artifact from the HRM code (which used Q-learning) -> in TRM, it's effectively a "Halting Classifier Loss."

![q_halt_loss.png](q_halt_loss.png)

3. **q_halt_accuracy**:
   This measures how good the model is at knowing it is finished.
   High q_halt_accuracy means the model correctly realizes: "I haven't solved this Sudoku yet, I need to recurse more" or "I have solved it, I should stop."

![q_halt_accuracy.png](q_halt_accuracy.png)

4. **accuracy (Cell-wise Accuracy)**:
   This one tripped me up a little - it's the percentage of individual(!) empty cells filled correctly.
   If a Sudoku has 50 empty cells and the model fills in 45 correctly, the accuracy is 90% (even though the Sudoku itself would be considered unsolved/wrong - therefore we need another metric here.

![accuracy](accuracy.png)

5. 