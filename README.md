# A simple computational model of mere exposure effects

This repository contains a small Python app to illustrate how a simple model of aesthetic value (Brielmann & Dayan, in prep.) can account for mere exposure effects.

You can see it in action [here](https://simple-mere-exposure-model.herokuapp.com/)

The current app lets you choose from a (somewhat) restricted range of parameter values and displays predicted aesthetic value over a given range of repetitions of the same (average) stimulus. 

In its current version, the app default is set to display the parameter settings that best fit the meta-analytic curve reported by: Montoya, R. M., Horton, R. S., Vevea, J. L., Citkowicz, M., & Lauber, E. A. (2017). A re-examination of the mere exposure effect: The influence of repeated exposure on recognition, familiarity, and liking. Psychological bulletin, 143(5), 459.

Current simplifying assumptions:

- representation in a 2D feature space
- covariance matrices are: stable, symmetric, contain equal variances for both features, and have 0 covariance
- learning rate is stable

 
---

 
Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
