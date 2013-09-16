%% optgrowth.m
% Alex Olssen
% 23/11/2012
% collocation solution to the continuous stochastic optimal growth model
% This is a matlab version of a python program that produces figure 3 of
% the paper by Jeno Pal and John Stachurski to be published in the
% Journal of Economic Dynamics & Control 37 (2013).
clear all; close all;
%% compecon
% I use Miranda and Fackler's compecon toolbox for Gaussian quadrature
% routines and Chebyshev polynomial routines.  It can be downloaded freely
% from www4.ncsu.edu/~pfackler/compecon/toolbox.html.  You need to set the
% toolbox path appropriately.
cepath='c:\compecon\'; path([cepath 'cetools;' cepath 'cedemos'],path);
%% use chebyshev polynomials for interpolation
smin   = 1e-5;                           % minimum of poly approx grid
smax   = 1;                              % maximum of poly approx grid
ssteps = 150;                            % plot step size
grid   = linspace(smin,smax,ssteps)';    % the grid for chebyshev reg
n      = 10;                             % max order chebyshev polys
fspace = fundefn('cheb',n+1,smin,smax);  % setup collaction
s      = funnode(fspace);                % collocation nodes
Phi    = funbas(fspace);                 % collocation matrix
%% parameters
alpha  = 0.33;                           % technology parameter
beta   = 0.95;                           % discount factor
sigma  = 0.25;                           % std. dev. of shocks
[e, w] = qnwlege(5, 1e-5, 4);            % setup Gaussian quadrature
%% initial guess
v = @(x) log(x);                         % utility fn
figure(1);
plot(grid, v(grid), 'b');
hold on;
%% function value iteration
tic
maxit = 7;  % this takes 21s on my pc
for t=1:maxit
  [vn, xn] = vmax(v, grid, e, w, alpha, beta, sigma);  % Update value fn
  c = funfitxy(fspace, grid, vn);  % Chebyshev regression on 150 nodes
  plot(grid, funeval(c, fspace, grid), 'b');
  v = @(x) funeval(c, fspace, x);  % Chebyshev interpolation
end
toc
%% plot true value function
theta = alpha * beta;
gamma = log(1 - theta) + theta * log(theta) / (1 - theta);
vtrue = log(grid) / (1 - theta) + gamma / (1 - beta);
plot(grid, vtrue, 'b--', 'linewidth', 2);
saveas(1, 'fig3.eps', 'eps');