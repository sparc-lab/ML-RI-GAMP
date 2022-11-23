The code is to reproduce the numerical results in *Approximate Message Passing for Multi-Layer Estimation in Rotationally Invariant Models*.

ML_RI_GAMP.m is the code for a 2-layer network with ReLU activation, Gaussian prior, Gaussian observation and Gaussian or Beta spectrum, and ML_RI_GAMP.m generates its SE.

Some parts of the code including free_cum_calc.m that calculates free cumulants of weight matrices are from Marco Mondelli's implementation of his work *Estimation in Rotationally Invariant Generalized Linear Models via Approximate Message Passing*. Some parts are from  https://github.com/GAMPTeam/vampyre.

