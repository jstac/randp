
.. _randp:

******************************************************************************
Fitted Value Function Iteration with Probability One Contractions
******************************************************************************

This page collects files and computer code for the paper **Fitted Value
Function Iteration with Probability One Contractions** by Jeno Pal and John
Stachurski.

Publication Details
-----------------------

| Fitted Value Function Iteration with Probability One Contractions
| Jeno Pal and John Stachurski
| **Journal of Economic Dynamics and Control**, 37 (2013) 251–264.


Abstract
-----------

This paper studies a value function iteration algorithm based on
nonexpansive function approximation and Monte Carlo integration that can
be applied to almost all stationary dynamic programming problems.  The
method can be represented using a randomized fitted Bellman operator and a
corresponding algorithm that is shown to be globally convergent with
probability one.  When additional restrictions are imposed, an
:math:`O_P(n^{-1/2})` rate of convergence for Monte Carlo error is obtained.


Code
--------

Python code can be found in the repository, and also on the `Google code site
<https://sites.google.com/site/fviprobone/home>`_.  Matlab code is posted
as well.  The latter is written by my RA Alex Olssen, and replicates figure 3 in
the paper.  There are two files, and the content is well documented and
self-explanitory.

Some general comments on coding are as follows: First, the paper suggests
kernel smoothers as one possible nonexpansive approximation method, and linear
interpolation as another.  In subsequent experiments, we've found that kernel
smoothers are usually a waste of time.  Linear interpolation is almost always
easier and better.  A second comment is that if you are working in a low
dimensional setting and find that Gaussian quadrature is more efficient than
Monte Carlo, there's no problems with using it:  Gaussian quadrature paired
with a nonexpansive approximator should again be nonexpansive.

