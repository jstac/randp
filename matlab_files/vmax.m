function [vn, xn] = vmax(v, s, e, w, alpha, beta, sigma)
  % VMAX evaluates the rhs of the bellman equation for the growth model.
  % Numerical integration is done by gaussian quadrature.
  % args
  %   v:       current guess for the value function
  %   s:       nodes where the current value function is evaluated
  %   e:       evaluation points for the discretized shock (quadrature)
  %   w:       weights for the discretized shock (quadrature)
  %   alpha:   technology parameter
  %   beta:    discount factor
  %   sigma:   variance of the shock
  n = length(s);
  vn = zeros(length(s), 1);
  xn = zeros(length(s), 1);
  for i=1:n
    maximand = @(x) valfn(x, v, s(i), e, w, alpha, beta, sigma);
    xn(i) = fminbnd(@(x) -1 * maximand(x), 0, s(i));
    vn(i) = maximand(xn(i));
  end
% evaluation of the maximand
function val = valfn(x, v, s, e, w, alpha, beta, sigma)
  Ev=0;
  K = length(e);
  for k=1:K
    vn = v(g(x, e(k), alpha)) * lognpdf(e(k), sigma);
    Ev = Ev + w(k) * vn;
  end
  val = f(s, x) + beta * Ev;
% reward function
function u = f(s, x)
  u = log(s - x);
% transition function
function s = g(x, e, alpha)
  s = e * x .^ alpha;
% lognormal pdf
function p = lognpdf(x, sigma)
  p = 1./(x.*sqrt(2*pi*sigma^2)).*exp(-log(x).^2/(2*sigma^2));