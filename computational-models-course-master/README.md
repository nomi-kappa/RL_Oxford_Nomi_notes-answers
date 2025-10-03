**This repository contains four tutorials on fitting computational models of learning to data from a study of reward-guided decision making.**

**The tutorials were originally developed for teaching to final year undergraduate students at the University of Oxford, by Laurence Hunt, Nils Kolling, Miriam Klein-Flugge and Jacqueline Scholl.**

In the first tutorial (developed by Laurence), students play a task that is based around the paradigm of Behrens et al., 2007, but it has an additional (between-subjects) stress manipulation. This paradigm runs in Psychtoolbox. The stress manipulation can be easily switched off, by setting trialvariables.playSound to be 0 for all trials (or by simply muting the sound).

In the second tutorial (developed by Nils), students learn about the basics of reinforcement learning, and the importance of simulating data from a model and parameter recovery from simulated data.

In the third tutorial (developed by Miriam), students fit a simple reinforcement learning model to the data to see if there are effects of stress and volatility on subjects' learning rates. **Note that for simplicity, the model fitted in this tutorial does not contain one of the parameters in Behrens et al., 2007, which distorts the probability function.** This free parameter is important for modelling the task, as it allows for subjects who assign different weights to reward probabilities and reward magnitudes.

In the final tutorial (developed by Jacquie), students learn about how to perform statistical inference on their fitted models, and also prepare to write up the practical.

Please let the tutorial authors know if you encounter any unintentional errors. Please also feel free to contact us if you have any questions, although we may not be able to respond to all enquiries.