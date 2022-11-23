% Compute the first nmax rectangular free cumulants when \Lambda is
% \sqrt{6} Beta(1, 2)

clear;
close all;
clc;

delta = 0.7;

nmax = 60;


m = zeros(1, nmax);

for i = 1 : nmax
    m(i) = 6^i/((i+1)*(2*i+1));
    
    if delta > 1
        m(i) = m(i)/delta;
    end
    
end

A = sym('A',[nmax nmax]);

syms x


for j = 1 : nmax 
    
    fprintf('%d\n', j);

    M = 0;

    for i = 1 : nmax +1 - j
        M = M + m(i)*x^i;
    end

    P = (x * (delta*M+1) * (M+1))^j;

    c = coeffs(P, 'All');
    
    for i = 1 : nmax
        A(j, i) = c(length(c)+1-i-1);
    end
    
end

k = sym('k',[nmax 1]);

k(1) = m(1);

for i = 2 : nmax
    k(i) = 0;
end
    
for i = 2 : nmax
    k(i) = m(i);
    
    for j = 1 : i-1
        k(i) = k(i) - k(j) * A(j, i);
    end
    
    
end

kdouble = double(k);