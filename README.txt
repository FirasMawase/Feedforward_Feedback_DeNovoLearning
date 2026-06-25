Feedforward-Feedback dynamic shapes de novo motor learning
============================================================
This folder contains a cleaned MATLAB simulation of 2D tracking during baseline, mirror-reversal learning, and washout.


Files
-----
- FeedbackFeedforwardDeNovoLearning2026.m - main MATLAB script with all local helper functions.

Quick Start
-----------
1. Open MATLAB.
2. Change the current folder to this directory.
3. Run:
    FeedbackFeedforwardDeNovoLearning2026.m

For a quick test run, set nSubjects = 2 near the top of the script. The full default run uses 10 simulated subjects and can take substantially longer.


What The Script Simulates
-------------------------
The model combines:
- a cortical feedforward recurrent neural network that receives target position, target velocity, and task context;
- a cerebellar-like feedback network driven by delayed visual and proprioceptive prediction errors;
- a second-order 2D arm plant;
- a mirror-reversal learning block followed by washout.

The script reports tracking error, frequency-specific orthogonal gain, feedforward and feedback contributions, recurrent-weight diagnostics, and after-effects.


Main Settings (all other information can be found in the submitted paper)
-------------
- Time step: dt = 0.01 s
- Trial duration: 40 s
- Target trajectory: 2D sum of sinusoids
- Visual delay: 70 ms
- Proprioceptive delay: 30 ms
- Default subjects: 10
- Trial blocks: 720 pretraining, 100 baseline, 360 mirror-reversal learning, 30 washout

