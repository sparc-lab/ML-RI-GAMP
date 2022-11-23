% ML-RI-GAMP
clear;
close all;
clc;

GAUSS = false;

delta1 = 2;
delta2 = 1.3;
sigmawgrid = 0.2; % noise standard deviation

n1 = 1000; % length of the hidden layer1
m1 = floor(n1*delta1); % length of the hidden layer2
m2 = floor(m1*delta2); % length of the output layer
ntrials = 100; % number of Montecarlo trials
niter = 15; % number of iterations of ML-RI-GAMP

toleigMM = 10^(-5);
toleigOmega = 10^(-9);
flagMM = 0;
toldiff = 10^(-8);

% the first 60 rectangular free cumulants of the spectrum of the design
% matrix are pre-computed by and stored in mat files
max_it = niter+1;

if GAUSS
    freecum1 = zeros(1, 2*max_it);
    freecum2 = zeros(1, 2*max_it);
    freecum1(1) = 1/delta1;
    freecum2(1) = 1/delta2;
else
    if delta1==1
        load freerectcum_beta_60_delta1.mat;
    elseif delta1==0.5
        load freerectcum_beta_60_delta0_5.mat;
    elseif delta1==1.1
        load freerectcum_beta_60_delta1_1.mat;
    elseif delta1==1.3
        load freerectcum_beta_60_delta1_3.mat;
    elseif delta1==2
        load freerectcum_beta_60_delta2.mat;
    elseif delta1==1.7
        load freerectcum_beta_60_delta1_7.mat;
    elseif delta1==0.7
        load freerectcum_beta_60_delta0_7.mat;
    else
        error('delta1 error');
    end

    freecum1 = zeros(1, 2*max_it);

    if niter > 50
        freecum1(1:100) = kdouble';
    else
        freecum1 = kdouble(1:2*max_it)'; % free cumulants (starting from the 1st)    
    end
    
    if delta2==1
        load freerectcum_beta_60_delta1.mat;
    elseif delta2==1.1
        load freerectcum_beta_60_delta1_1.mat;
    elseif delta2==1.3
        load freerectcum_beta_60_delta1_3.mat;
    else
        error('delta2 error');
    end

    freecum2 = zeros(1, 2*max_it);

    if niter > 50
        freecum2(1:100) = kdouble';
    else
        freecum2 = kdouble(1:2*max_it)'; % free cumulants (starting from the 1st)    
    end
end

time = zeros(ntrials, length(sigmawgrid));

% contains *true* normalized squared correlations between the signal and 
% the x iterate produced by ML-RI-GAMP
scal_allx1 = zeros(niter, ntrials, length(sigmawgrid)); 

% contains normalized squared correlations between the signal and the x
% iterate produced by ML-RI-GAMP, as estimated from state evolution parameters
scal_estallx1 = zeros(niter, ntrials, length(sigmawgrid));

% contains *true* normalized squared correlations between the signal and
% the \hat{x} iterate produced by ML-RI-GAMP
scal_allhatx1 = zeros(niter, ntrials, length(sigmawgrid)); 

% contains normalized squared correlations between the signal and the
% \hat{x} iterate produced by ML-RI-GAMP, as estimated from state evolution
% parameters
scal_estallhatx1 = zeros(niter, ntrials, length(sigmawgrid));

% for y1
scal_allx2 = zeros(niter, ntrials, length(sigmawgrid)); 
scal_estallx2 = zeros(niter, ntrials, length(sigmawgrid));
scal_allhatx2 = zeros(niter, ntrials, length(sigmawgrid)); 
scal_estallhatx2 = zeros(niter, ntrials, length(sigmawgrid));

for j = 1 : length(sigmawgrid)  
    
    sigmaw = sigmawgrid(j);

    for i = 1 : ntrials
        fprintf('trial #%d\n', i);

        % The signal x has a Gaussian prior
        x = randn(n1, 1);
        
        if GAUSS
            A1 = randn(m1,n1)/sqrt(m1);
            y1 = max(A1*x,0);
            A2 = randn(m2,m1)/sqrt(m2);
            y2 = A2 * y1 + sigmaw * randn(m2, 1);
        else
            % The design matrix A1 is equal to U*Lambda*V', where U, V are Haar 
            % distributed (obtained from the SVD of a Gaussian matrix) and 
            % Lambda has i.i.d. \sqrt{6}*Beta(1, 2) diagonal entries
            if delta1 <= 1
                Lambda = sqrt(6)*[diag(betarnd(ones(m1, 1), 2*ones(m1, 1))), zeros(m1, n1-m1)];
            else
                Lambda = sqrt(6)*[diag(betarnd(ones(n1, 1), 2*ones(n1, 1))); zeros(m1-n1, n1)];
            end            

            G = randn(m1,n1);

            [U, ~, V] = svd(G);

            A1 = U * Lambda * V';

            y1 = max(A1*x,0);

            % The design matrix A2
            if delta2 <= 1
                Lambda = sqrt(6)*[diag(betarnd(ones(m2, 1), 2*ones(m2, 1))), zeros(m2, m1-m2)];
            else
                Lambda = sqrt(6)*[diag(betarnd(ones(m1, 1), 2*ones(m1, 1))); zeros(m2-m1, m1)];
            end            

            G = randn(m2,m1);

            [U, ~, V] = svd(G);

            A2 = U * Lambda * V';

            y2 = A2 * y1 + sigmaw * randn(m2, 1);
        end
        % approximates E[Y^2]
        expys = sum(y2.^2)/m2;
                
        % allocate vectors for ML-RI-GAMP iterations1
        xAMP1 = zeros(n1, niter);
        xhatAMP1 = zeros(n1, niter);
        sAMP1 = zeros(m1, niter+1);
        rAMP1 = zeros(m1, niter);

        % allocate vectors for SE computations (needed to compute the
        % denoisers in ML-RI-GAMP)
        muSE1 = zeros(niter, 1);
        sigmaSE1 = zeros(niter, 1);
        scalx1 = zeros(niter, 1);
        scalhatx1 = zeros(niter, 1);
        PsiAUX1 = zeros(niter+1, niter+1);
        PhiAUX1 = zeros(niter+2, niter+2);
        Omega1 = zeros(niter, niter);
        Sigma1 = zeros(niter+1, niter+1);
        DeltaAUX1 = zeros(niter+2, niter+2);
        GammaAUX1 = zeros(niter+1, niter+1);
        GammaAUX2 = zeros(niter+1, niter+1);
        
        % initialization of the SE quantities
        rho = 0.5; % after relu, rho part becomes zero
        GammaAUX1(1, 1) = 1;
        PhiAUX1(2, 1) = rho * delta2 * freecum2(1);
        
        PsiAUXtmp1 = PsiAUX1(1:2, 1:2);
        PhiAUXtmp1 = PhiAUX1(1:2, 1:2);
        
        S1 = PsiAUXtmp1 * PhiAUXtmp1;
        
       %% layer1
        sAMP1(:, 1) = A2' * y2;
        xAMP1(:, 1) = A1' * sAMP1(:, 1);
        DeltaAUX1(2, 2) = mean(sAMP1(:, 1).^2);
        DeltaAUXtmp1 = DeltaAUX1(1:2, 1:2);

        % initialization of the SE quantities \mu_1, \Sigma_1 and
        % \Omega_1 (the last one is estimated from the data)
        muSE1(1) = 0;

        for jj = 0 : 1
            Mattmp = PhiAUXtmp1 * S1^jj;
            muSE1(1) = muSE1(1) + delta1 * freecum1(jj+1) * Mattmp(2, 1);
        end        
        sigmaSE1(1) = mean(xAMP1(:, 1).^2) - muSE1(1)^2;
        
        Omega1(1, 1) = sigmaSE1(1);
        
        % \Omega_1 is a variance, so it should be non-negative
        if sigmaSE1(1) < 0
            fprintf('Something is strange, sigma(%d)=%f\n', 1, sigmaSE1(1));
            sigmaSE1(1) = abs(sigmaSE1(1));
        end
        
        % computation of \Psi_2
        PsiAUX1(2, 2) = muSE1(1)/(sigmaSE1(1)+muSE1(1)^2);
                
        % computation of ML-RI-GAMP iterates \hat{x}^1 and r^1
        xhatAMP1(:, 1) = muSE1(1) * xAMP1(:, 1) / (sigmaSE1(1)+muSE1(1)^2);  
        rAMP1(:, 1) = A1 * xhatAMP1(:, 1) - freecum1(1) * PsiAUX1(2, 2) * sAMP1(:, 1);
        
        % estimation of \Gamma_2 from the data 
        GammaAUX1(1, 2) = mean(xhatAMP1(:, 1).^2);
        GammaAUX1(2, 1) = GammaAUX1(1, 2);
        GammaAUX1(2, 2) = GammaAUX1(1, 2);
        
        % computation of \Sigma_2 from the SE recursion
        GammaAUXtmp1 = GammaAUX1(1:2, 1:2);
        PsiAUXtmp1 = PsiAUX1(1:2, 1:2);

        S1 = PsiAUXtmp1 * PhiAUXtmp1;

        SigmaSEtmp1 = zeros(2, 2);

        for t1 = 0 : 3
            pow = zeros(2, 2);
            for t2 = 0 : t1
                pow = pow + S1^t2 * GammaAUXtmp1 * (S1')^(t1-t2);
            end                   
            pow2 = zeros(2, 2);
            for t2 = 0 : t1-1
                pow2 = pow2 + S1^t2 * PsiAUXtmp1 * DeltaAUXtmp1 * PsiAUXtmp1' * (S1')^(t1-t2-1);
            end
            SigmaSEtmp1 = SigmaSEtmp1 + freecum1(t1+1) * (pow + pow2);
        end

        Sigma1(1:2, 1:2) = SigmaSEtmp1;
        
        % to do the denoising of relu activation
        sigmat1 = Sigma1(1, 1) - Sigma1(1, 2)/ Sigma1(2, 2) * Sigma1(2, 1);
        rhat1 = Sigma1(1, 2)/ Sigma1(2, 2) * rAMP1(:, 1);
       %% layer2
       % allocate vectors for RI-GAMP iterations
        xAMP2 = zeros(m1, niter);
        xhatAMP2 = zeros(m1, niter);
        sAMP2 = zeros(m2, niter+1);
        rAMP2 = zeros(m2, niter);

        % allocate vectors for SE computations (needed to compute the
        % denoisers in RI-GAMP)
        muSE2 = zeros(niter, 1);
        sigmaSE2 = zeros(niter, 1);
        scalx2 = zeros(niter, 1);
        scalhatx2 = zeros(niter, 1);
        PsiAUX2 = zeros(niter+1, niter+1);
        PhiAUX2 = zeros(niter+2, niter+2);
        Omega2 = zeros(niter, niter);
        Sigma2 = zeros(niter+1, niter+1);
        DeltaAUX2 = zeros(niter+2, niter+2);

        % initialization of the SE quantity
        PhiAUX2(2, 1) = 1;
        GammaAUX2(1, 1) = rho*freecum1(1);
        
       % initialization of ML-RI-GAMP
        sAMP2(:, 1) = y2;        
        xAMP2(:, 1) = A2' * sAMP2(:, 1);

        % initialization of the SE quantities \mu_1, \Sigma_1 and
        % \Omega_1 (the last one is estimated from the data)
        muSE2(1) = delta2 * freecum2(1);      
        sigmaSE2(1) = mean(xAMP2(:, 1).^2) - muSE2(1)^2*GammaAUX2(1, 1);
        Omega2(1, 1) = sigmaSE2(1);
        Sigma2(1, 1) = freecum2(1)*GammaAUX2(1, 1);
        
        % \Omega_1 is a variance, so it should be non-negative
        if sigmaSE2(1) < 0
            fprintf('Something is strange, sigma(%d)=%f\n', 1, sigmaSE2(1));
            sigmaSE2(1) = abs(sigmaSE2(1));
        end
        
        % denoising for relu
        xt2 = xAMP2(:, 1) / muSE2(1);
        rhot2 = sigmaSE2(1) / muSE2(1)^2;
        try
            [zhat0,zhat1,zhatvar0,zhatvar1] = est_mmse(rhat1,xt2,sigmat1,rhot2);
        catch
            disp('numerical error in estimation!');
            break;
        end

        xhatAMP2(:, 1) = zhat1;
        % computation of \Psi_2
        PsiAUX2(2, 2) = zhatvar1/rhot2 / muSE2(1);
        
        % computation of ML-RI-GAMP iterates r^1
        rAMP2(:, 1) = A2 * xhatAMP2(:, 1) - freecum2(1) * PsiAUX2(2, 2) * sAMP2(:, 1);
       %% computation of the RI-GAMP iterate s^2 
       %% layer 1
        % denoising for relu
        sAMP1(:, 2) = zhat0 - rhat1;
        PhiAUX1(3, 2) =  Sigma1(1, 2)/ Sigma1(2, 2) * (zhatvar0/sigmat1-1);
        % estimation of \Phi_3 from data  
        PhiAUX1(3, 1) = mean(sAMP1(:, 2).^2)/sigmat1;
        
        % estimation of \Delta_3 from the data
        DeltaAUX1(3, 3) = mean(sAMP1(:, 2).^2);
        DeltaAUX1(2, 3) = mean(sAMP1(:, 2).*sAMP1(:, 1));
        DeltaAUX1(3, 2) = DeltaAUX1(2, 3);
        
        fprintf('layer 1\n');
        % computation of true and estimated normalized squared correlations
        % for x^1 and \hat{x}^1
        scalx1(1) = (sum(xAMP1(:, 1).* x))^2/sum(x.^2)/sum(xAMP1(:, 1).^2);
        scal_allx1(1, i ,j) = scalx1(1);
        scal_estallx1(1, i, j) = muSE1(1)^2/(muSE1(1)^2+sigmaSE1(1));
        scalhatx1(1) = (sum(xhatAMP1(:, 1).* x))^2/sum(x.^2)/sum(xhatAMP1(:, 1).^2);
        scal_allhatx1(1, i ,j) = scalhatx1(1);
        scal_estallhatx1(1, i, j) = mean(xhatAMP1(:, 1).^2);
        fprintf('Iteration %d, True squared correlation=%f, Estimated squared correlation=%f\n', ...
            1, scalx1(1), scal_estallx1(1, i, j));
        fprintf('hat-iterate: True squared correlation=%f, Estimated squared correlation=%f\n', ...
            scalhatx1(1), scal_estallhatx1(1, i, j));

       %% layer 2
        % estimation of \Sigma_2 from the data         
        Sigma2(2, 2) = mean(rAMP2(:, 1).*rAMP2(:, 1));
        Sigma2(1, 2) = mean(rAMP2(:, 1).*y2);
        Sigma2(2, 1) = Sigma2(1, 2); 
        
        MM = [Sigma2(1:2, 1:2), Sigma2(1:2, 1); Sigma2(1, 1:2), expys]; 
        
        % the matrix MM(2:3, 2:3) should be PSD and hence have non-negative
        % determinant
        if det(MM(2:3, 2:3)) < 0 
            fprintf('det(MM)<0 at the beginning: Numerical precision exceeded! Warning!\n');
        end
        
        auxMM = MM(1, 2:3) / MM(2:3, 2:3);

        % computation of the ML-RI-GAMP iterate s^2  
        sAMP2(:, 2) = ( auxMM * [rAMP2(:, 1), y2]')';
        
        % Sigma(2, 2) is a variance, so it should be non-negative
        if Sigma2(2, 2) < 0 
            fprintf('det(Sigma)<0 at the beginning: Numerical precision exceeded! Warning!\n');
        end
        
        auxMM2 = Sigma2(1, 2) / Sigma2(2, 2);
        sAMP2(:, 2) = sAMP2(:, 2) - auxMM2 * rAMP2(:, 1);
        
        % computation of \Phi_3 from SE parameters  
        PhiAUX2(3, 1) = auxMM(2);
        PhiAUX2(3, 2) = auxMM(1)-auxMM2;
        
        % estimation of \Gamma_2 and \Delta_3 from the data
        GammaAUX2(1, 2) = mean(xhatAMP2(:, 1).^2);
        GammaAUX2(2, 1) = GammaAUX2(1, 2);
        GammaAUX2(2, 2) = GammaAUX2(1, 2);
           
        DeltaAUX2(2, 2) = mean( sAMP2(:, 1).^2 );
        DeltaAUX2(3, 3) = mean( sAMP2(:, 2).^2 );
        DeltaAUX2(2, 3) = mean( sAMP2(:, 2).*sAMP2(:, 1) );
        DeltaAUX2(3, 2) = DeltaAUX2(2, 3);
       
        fprintf('layer 2\n');
        % computation of true and estimated normalized squared correlations
        % for x^1 and \hat{x}^1
        scalx2(1) = (sum(xAMP2(:, 1).* y1))^2/sum(y1.^2)/sum(xAMP2(:, 1).^2);
        scal_estallx2(1, i, j) = muSE2(1)^2*GammaAUX2(1, 1)/(muSE2(1)^2*GammaAUX2(1, 1)+sigmaSE2(1));
        scalhatx2(1) = (mean(xhatAMP2(:, 1).* y1))^2/mean(y1.^2)/mean(xhatAMP2(:, 1).^2);
        scal_estallhatx2(1, i, j) = sum(xhatAMP2(:, 1).^2)/m1/GammaAUX2(1, 1);
        fprintf('Iteration %d, True squared correlation=%f, Estimated squared correlation=%f\n', ...
            1, scalx2(1), scal_estallx2(1, i, j));
        fprintf('hat-iterate: True squared correlation=%f, Estimated squared correlation=%f\n', ...
            scalhatx2(1), scal_estallhatx2(1, i, j));
        
        flagMM = 0;
        %% iteration
        for jj = 2 : niter
           %% layer 1
           fprintf('layer 1\n');
           % computation of the Onsager coefficients needed by ML-RI-GAMP
           PsiAUXtmp1 = PsiAUX1(1:jj+1, 1:jj+1);
           PhiAUXtmp1 = PhiAUX1(1:jj+1, 1:jj+1); 
           DeltaAUXtmp1 = DeltaAUX1(1:jj+1, 1:jj+1);
           GammaAUXtmp1 = GammaAUX1(1:jj+1, 1:jj+1);
           S1 = PsiAUXtmp1 * PhiAUXtmp1;
           
           Mattmp = zeros(jj+1, jj+1);
           for j2 = 0 : jj        
               Mattmp = Mattmp + delta1 * freecum1(j2+1) * PhiAUXtmp1 * S1^j2;
           end
           
           beta2 = Mattmp(jj+1, 2:jj);
           
           % computation of the RI-GAMP iterate x^{jj} 
           xAMP1(:, jj) = A1' * sAMP1(:, jj) - sum(repmat(beta2, n1, 1) .* xhatAMP1(:, 1:jj-1), 2);
                    
           Q = PhiAUXtmp1 * PsiAUXtmp1;

           % computation of \Omega_{jj} from the SE recursion
           OmegaSEtmp1 = zeros(jj+1, jj+1);
           for t1 = 0 : 2*jj
               pow = zeros(jj+1, jj+1);
               for t2 = 0 : t1
                   pow = pow + Q^t2 * DeltaAUXtmp1 * (Q')^(t1-t2);
               end
               pow2 = zeros(jj+1, jj+1);
               for t2 = 0 : t1-1
                   pow2 = pow2 + Q^t2 * PhiAUXtmp1 * GammaAUXtmp1 * PhiAUXtmp1' * (Q')^(t1-t2-1);
               end
               OmegaSEtmp1 = OmegaSEtmp1 + freecum1(t1+1) * (pow + pow2);
           end

           OmegaSEtmp1 = delta1 * OmegaSEtmp1;
   
           Omega1(1:jj, 1:jj) = OmegaSEtmp1(2:jj+1, 2:jj+1);
           sigmaSE1(jj) = Omega1(jj, jj);
           
           % estimate \mu from data
           muSE1(jj) = sqrt(mean(xAMP1(:, jj).^2)-sigmaSE1(jj));
                   
           % computation of true and estimated normalized squared
           % correlation for x^{jj}           
           scalx1(jj) = (sum(xAMP1(:, jj).* x))^2/sum(x.^2)/sum(xAMP1(:, jj).^2);
           scal_allx1(jj, i ,j) = scalx1(jj);
           scal_estallx1(jj, i, j) = muSE1(jj)^2/(muSE1(jj)^2+sigmaSE1(jj));
           
           fprintf('Iteration %d, True squared correlation=%f, Estimated squared correlation=%f\n', ...
               jj, scalx1(jj), muSE1(jj)^2/(muSE1(jj)^2+sigmaSE1(jj)));
           
           % the matrix \Omega_{jj} should be PSD
           Omegared1 = Omega1(1:jj, 1:jj);
           
           if min(eig(Omegared1)) < toleigOmega 
               fprintf('min(eig(Omega))) too small: Numerical precision exceeded! Warning!\n');
               min(eig(Omegared1))
               break;
           end
           
           % computation of the ML-RI-GAMP iterate \hat{x}^{jj} 
           xhatAMP1(:, jj) = (muSE1(1:jj)' / (Omegared1 + muSE1(1:jj) * muSE1(1:jj)') * xAMP1(:, 1:jj)')';

           % computation of true and estimated normalized squared
           % correlation for \hat{x}^{jj}
           scalhatx1(jj) = (sum(xhatAMP1(:, jj).* x))^2/sum(x.^2)/sum(xhatAMP1(:, jj).^2);
           
           % early stop
           if jj>2 && sum(xhatAMP1(:, jj).^2)/n1<scal_estallhatx1(jj-1, i, j)
               disp('Correlation becomes smaller, stop!')
               break;
           end
           
           if jj>2 && scalhatx1(jj)<scalhatx1(jj-1)
               disp('Correlation becomes smaller, stop!')
               break;
           end
           
           scal_allhatx1(jj, i ,j) = scalhatx1(jj);
           scal_estallhatx1(jj, i, j) = mean(xhatAMP1(:, jj).^2);
           fprintf('hat-iterate: True squared correlation=%f, Estimated squared correlation=%f\n', ...
               scalhatx1(jj), scal_estallhatx1(jj, i, j));
           
           
           % computation of \Psi_{jj+1} from SE parameters  
           PsiAUX1(jj+1, 2:jj+1) = muSE1(1:jj)' / (Omegared1 + muSE1(1:jj) * muSE1(1:jj)');
           PsiAUXtmp1 = PsiAUX1(1:jj+1, 1:jj+1);
           
           Q = PhiAUXtmp1 * PsiAUXtmp1;
           
           Mattmp = zeros(jj+1, jj+1);
    
           for j2 = 0 : jj+1        
               Mattmp = Mattmp + freecum1(j2+1) * PsiAUXtmp1 * Q^j2;
           end
           
           % computation of the Onsager coefficients needed by RI-GAMP
           alpha = Mattmp(jj+1, 2:jj+1);
           
           % computation of the ML-RI-GAMP iterate r^{jj} 
           rAMP1(:, jj) = A1 * xhatAMP1(:, jj) - sum(repmat(alpha, m1, 1) .* sAMP1(:, 1:jj), 2);
           
           % estimation of \Gamma_{jj+1} from the data
           GammaAUX1(1, jj+1) = mean(xhatAMP1(:, jj).^2);
           GammaAUX1(jj+1, 1) = GammaAUX1(1, jj+1);
           GammaAUX1(jj+1, jj+1) = GammaAUX1(1, jj+1);  
            
           for j2 = 2 : jj          
               GammaAUX1(j2, jj+1) = mean(xhatAMP1(:, jj).* xhatAMP1(:, j2-1));
               GammaAUX1(jj+1, j2) = GammaAUX1(j2, jj+1);
           end
               
           GammaAUXtmp1 = GammaAUX1(1:jj+1, 1:jj+1);
           
           % computation of \Sigma_{jj+1} from the SE recursion
           PsiAUXtmp1 = PsiAUX1(1:jj+1, 1:jj+1);
           S1 = PsiAUXtmp1 * PhiAUXtmp1;
           SigmaSEtmp1 = zeros(jj+1, jj+1);
           for t1 = 0 : 2*jj+1
                pow = zeros(jj+1, jj+1);
                for t2 = 0 : t1
                    pow = pow + S1^t2 * GammaAUXtmp1 * (S1')^(t1-t2);
                end
                pow2 = zeros(jj+1, jj+1);
                for t2 = 0 : t1-1
                    pow2 = pow2 + S1^t2 * PsiAUXtmp1 * DeltaAUXtmp1 * PsiAUXtmp1' * (S1')^(t1-t2-1);
                end
                SigmaSEtmp1 = SigmaSEtmp1 + freecum1(t1+1) * (pow + pow2);
            end

           Sigma1(1:jj+1, 1:jj+1) = SigmaSEtmp1;
           
           % the matrix Sigma(2:jj+1, 2:jj+1) should be PSD
           if min(eig(Sigma1(2:jj+1, 2:jj+1))) < 0 
               fprintf('min(eig(Sigma))<0: Numerical precision exceeded! Warning!\n');
               min(eig(Sigma1(2:jj+1, 2:jj+1)))
               break;
           end
                     
           % for denoising of relu activation
           sigmat1 = Sigma1(1, 1) - Sigma1(1, 2:jj+1) / Sigma1(2:jj+1, 2:jj+1) * Sigma1(2:jj+1, 1);

           if sigmat1 < 0 
               fprintf('sigmat(%d)=%f<0: Numerical precision exceeded! Warning!\n', jj, sigmat1);
           end
           
           sigmat1 = abs(sigmat1);
           rhat1 = (Sigma1(1, 2:jj+1) / Sigma1(2:jj+1, 2:jj+1) * (rAMP1(:, 1:jj))')';
           
           %% layer 2
           fprintf('layer 2\n');
           % computation of \mu_{jj} from the SE recursion
           PsiAUXtmp2 = PsiAUX2(1:jj+1, 1:jj+1);
           PhiAUXtmp2 = PhiAUX2(1:jj+1, 1:jj+1); 
           DeltaAUXtmp2 = DeltaAUX2(1:jj+1, 1:jj+1);
           GammaAUXtmp2 = GammaAUX2(1:jj+1, 1:jj+1);
           S2 = PsiAUXtmp2 * PhiAUXtmp2;
           
           Mattmp = zeros(jj+1, jj+1);
    
           for j2 = 0 : jj        
               Mattmp = Mattmp + delta2 * freecum2(j2+1) * PhiAUXtmp2 * S2^j2;
           end
           
           muSE2(jj) = Mattmp(jj+1, 1);
           
           % computation of the Onsager coefficients needed by RI-GAMP
           beta2 = Mattmp(jj+1, 2:jj);
           
           % computation of the RI-GAMP iterate x^{jj} 
           xAMP2(:, jj) = A2' * sAMP2(:, jj) - sum(repmat(beta2, m1, 1) .* xhatAMP2(:, 1:jj-1), 2);
                   
           % computation of \Omega_{jj} from the SE recursion
           Q = PhiAUXtmp2 * PsiAUXtmp2;
           OmegaSEtmp2 = zeros(jj+1, jj+1);

           for t1 = 0 : 2*jj
               pow = zeros(jj+1, jj+1);

               for t2 = 0 : t1
                   pow = pow + Q^t2 * DeltaAUXtmp2 * (Q')^(t1-t2);
               end

               pow2 = zeros(jj+1, jj+1);

               for t2 = 0 : t1-1
                   pow2 = pow2 + Q^t2 * PhiAUXtmp2 * GammaAUXtmp2 * PhiAUXtmp2' * (Q')^(t1-t2-1);
               end

               OmegaSEtmp2 = OmegaSEtmp2 + freecum2(t1+1) * (pow + pow2);
           end

           OmegaSEtmp2 = delta2 * OmegaSEtmp2;
   
           Omega2(1:jj, 1:jj) = OmegaSEtmp2(2:jj+1, 2:jj+1);
           sigmaSE2(jj) = Omega2(jj, jj);
           muSE2(jj) = sqrt(mean(xAMP2(:, jj).^2)-sigmaSE2(jj))/sqrt(GammaAUX2(1, 1));
           
           % computation of true and estimated normalized squared
           % correlation for x^{jj}
           scalx2(jj) = (sum(xAMP2(:, jj).* y1))^2/sum(y1.^2)/sum(xAMP2(:, jj).^2);
           
           scal_estallx2(jj, i, j) = muSE2(jj)^2*GammaAUX2(1, 1)/(muSE2(jj)^2*GammaAUX2(1, 1)+sigmaSE2(jj));
           
           fprintf('Iteration %d, True squared correlation=%f, Estimated squared correlation=%f\n', ...
               jj, scalx2(jj), scal_estallx2(jj, i, j));
           
           Omegared2 = Omega2(1:jj, 1:jj);
           
           % the matrix \Omega_{jj} should be PSD
           if min(eig(Omegared2)) < toleigOmega 
               fprintf('min(eig(Omega))) too small: Numerical precision exceeded! Warning!\n');
               min(eig(Omegared2))
               break;
           end
          
            % denoising of relu activation
            xt2 = (muSE2(1:jj)'/Omegared2*xAMP2(:,1:jj)' / (muSE2(1:jj)'/Omegared2*muSE2(1:jj)))';
            rhot2 = 1 / (muSE2(1:jj)'/Omegared2*muSE2(1:jj));
            try
                [zhat0,zhat1,zhatvar0,zhatvar1] = est_mmse(rhat1,xt2,sigmat1,rhot2);
            catch
                disp('numerical error!');
                break;
            end
            xhatAMP2(:, jj) = zhat1;
            PsiAUX2(jj+1, 2:jj+1) = zhatvar1/rhot2 * muSE2(1:jj)'/Omegared2*rhot2;   
            
           % computation of true and estimated normalized squared
           % correlation for \hat{x}^{jj}
           scalhatx2(jj) = (sum(xhatAMP2(:, jj).* y1))^2/sum(y1.^2)/sum(xhatAMP2(:, jj).^2);
           scal_estallhatx2(jj, i, j) = mean(xhatAMP2(:, jj).^2)/GammaAUX2(1, 1);
           fprintf('hat-iterate: True squared correlation=%f, Estimated squared correlation=%f\n', ...
               scalhatx2(jj), scal_estallhatx2(jj, i, j));
           
           PsiAUXtmp2 = PsiAUX2(1:jj+1, 1:jj+1);
           
           Q = PhiAUXtmp2 * PsiAUXtmp2;
           
           Mattmp2 = zeros(jj+1, jj+1);
    
           for j2 = 0 : jj+1        
               Mattmp2 = Mattmp2 + freecum2(j2+1) * PsiAUXtmp2 * Q^j2;
           end
           
           % computation of the Onsager coefficients needed by RI-GAMP
           alpha = Mattmp2(jj+1, 2:jj+1);
           
           % computation of the ML-RI-GAMP iterate r^{jj} 
           rAMP2(:, jj) = A2 * xhatAMP2(:, jj) - sum(repmat(alpha, m2, 1) .* sAMP2(:, 1:jj), 2);
           % estimation of \Gamma_{jj+1} from the data
           GammaAUX2(1, jj+1) = mean(xhatAMP2(:, jj).^2);
           GammaAUX2(jj+1, 1) = GammaAUX2(1, jj+1);
           GammaAUX2(jj+1, jj+1) = GammaAUX2(1, jj+1);  
            
           for j2 = 2 : jj          
               GammaAUX2(j2, jj+1) = mean(xhatAMP2(:, jj).* xhatAMP2(:, j2-1));
               GammaAUX2(jj+1, j2) = GammaAUX2(j2, jj+1);
           end
           % estimation of \Sigma_{jj+1} from data
           Sigma2(jj+1, jj+1) = mean(rAMP2(:, jj).*rAMP2(:, jj));
           
           for j2 = 2 : jj
               Sigma2(j2, jj+1) = mean(rAMP2(:, jj).*rAMP2(:, j2-1));
               Sigma2(jj+1, j2) = Sigma2(j2, jj+1);
           end
           Sigma2(1, jj+1) = mean(rAMP2(:, jj).*y2);
           Sigma2(jj+1, 1) = Sigma2(1, jj+1);
           
           % the matrix MM(2:jj+2, 2:jj+2) should be PSD
           MM = [Sigma2(1:jj+1, 1:jj+1), Sigma2(1:jj+1, 1); Sigma2(1, 1:jj+1), expys]; 
        
           
           if det(MM(2:end, 2:end)) < 0 
               fprintf('det(MM)=%f: Numerical precision exceeded! Warning!\n', det(MM(2:end, 2:end)));
               break;
           end
           %% calculate sAMP
           %% layer 1
           sAMP1(:, jj+1) = zhat0 - rhat1;
           
           % estimate \phi
           PhiAUX1(jj+2, 1) = mean(sAMP1(:, jj+1).^2)/sigmat1;
           
           % stop if the algorithm is unstable and some quantities are NaN
           flgNaN = 0;           
           for t2 = 2 : jj+1
               if isnan(PhiAUX1(jj+2, t2))
                   flgNaN = 1;
               end
           end
           
           if flgNaN == 1
               fprintf('NaN!\n');
               break;
           end
        
           if isnan(PhiAUX1(jj+2, 1)) || PhiAUX1(jj+2, 1) == Inf || PhiAUX1(jj+2, 1) == -Inf
               fprintf('NaN!\n');
               break;
           end
           
           hder = zhatvar0/sigmat1-1;

           PhiAUX1(jj+2, 2:jj+1) = hder * Sigma1(1, 2:jj+1) / Sigma1(2:jj+1, 2:jj+1);
               
           % estimation of \Delta_{jj+2} from the data
           DeltaAUX1(jj+2, jj+2) = mean(sAMP1(:, jj+1).^2);
            
           for j2 = 1 : jj          
               DeltaAUX1(j2+1, jj+2) = mean(sAMP1(:, jj+1).* sAMP1(:, j2));
               DeltaAUX1(jj+2, j2+1) = DeltaAUX1(j2+1, jj+2);
           end
        %% layer 2
           if (min(eig(MM(2:end, 2:end))) < toleigMM) || flagMM == 1
               % if MM is close to singular, we employ only a single memory
               % term for the denoiser h_t
               fprintf('MM is close to singular, now using 1 memory term only for ht\n');
               flagMM = 1;

               % computation of the ML-RI-GAMP iterate s^{jj+1}  
               sAMP2(:, jj+1) = -rAMP2(:, jj) + y2;
               
               % computation of \Phi_{jj+2} 
               PhiAUX2(jj+2, 1) = 1;
               PhiAUX2(jj+2, jj+1) = -1;
                 
           else
               auxMM = MM(1, 2:end) / MM(2:end, 2:end); 
               sAMP2(:, jj+1) = (auxMM * [rAMP2(:, 1:jj), y2]')';
               if min(eig(Sigma2(2:jj+1, 2:jj+1))) < 0 
                   fprintf('min(eig(Sigma))<0: Numerical precision exceeded! Warning!\n');
                   min(eig(Sigma2(2:jj+1, 2:jj+1)))
                   break;
               end

               auxMM2 = Sigma2(1, 2:jj+1) / Sigma2(2:jj+1, 2:jj+1);
               sAMP2(:, jj+1) = sAMP2(:, jj+1) - (auxMM2 * rAMP2(:, 1:jj)')';
               PhiAUX2(jj+2, 1) = auxMM(jj+1);
               for j2 = 1 : jj
                   PhiAUX2(jj+2, j2+1) = auxMM(j2)-auxMM2(j2);
               end 
           end
           
           % estimation of \Delta_{jj+2} from the data
           DeltaAUX2(jj+2, jj+2) = mean(sAMP2(:, jj+1).^2);
           for j2 = 1 : jj          
               DeltaAUX2(j2+1, jj+2) = mean(sAMP2(:, jj+1).* sAMP2(:, j2));
               DeltaAUX2(jj+2, j2+1) = DeltaAUX2(j2+1, jj+2);
           end
        end   
    end
end

function [z0,z1,z2]=gauss_integral(a,b,mu,var)
    % Get standard deviation
    sig = sqrt(var);
        
    if b == Inf
        % Integral from [a,infty]
        alpha = (a-mu)/sig;
        f = 1/(sqrt(2*pi))*exp(-alpha.^2/2);
        Q = 0.5*erfc(alpha/sqrt(2)) ;
        w0 = Q;
        w1 = f;
        w2 = Q+alpha.*f;
    elseif a == -Inf
        % Integral from [-infty,b]
        beta = (b-mu)/sig;
        f = 1/(sqrt(2*pi))*exp(-beta.^2/2);
        F = 0.5*(1+erf(beta/sqrt(2)));
        w0 = F;
        w1 = -f;
        w2 = F-beta.*f;      
     else
        error("Only single sided distributions handled for now");
     end
      
    % Convert back to z
    z0 = sig*w0;
    z1 = sig*(mu.*w0 + sig.*w1);
    z2 = sig*((mu.^2).*w0 + 2*mu*sig.*w1 + var*w2);
end

function [zhat0,zhat1,zhatvar0,zhatvar1]=est_mmse(r0,r1,rvar0,rvar1)

    % Compute the conditional Gaussian terms for z > 0 and z < 0
    zvarp = rvar0*rvar1/(rvar0+rvar1);
    zvarn = rvar0;
    rp = (rvar1*r0 + rvar0*r1)/(rvar0+rvar1);        
    rn = r0;

    % Compute scaling constants for each region
    Ap = 0.5*((rp.^2)/zvarp - (r0.^2)/rvar0 - (r1.^2)/rvar1);
    An = 0.5*(-(r1.^2)/rvar1);
    Amax = max(Ap,An);
    Ap = Ap - Amax;
    An = An - Amax;
    Cp = exp(Ap);
    Cn = exp(An);

    % Compute moments for each region
    [temp1,temp2,temp3] = gauss_integral(0, Inf, rp, zvarp);
    zp = Cp.*[temp1,temp2,temp3];
    [temp1,temp2,temp3] = gauss_integral(-Inf, 0, rn, zvarn);
    zn = Cn.*[temp1,temp2,temp3];
    
    zpsum = zp(:,1) + zn(:,1);

    % Compute mean        
    zhat0 = (zp(:,2) + zn(:,2))./zpsum;
    zhat1 = zp(:,2)./zpsum;

   % Compute the variance
    zhatvar0 = mean((zp(:,3) + zn(:,3))./zpsum - zhat0.^2);
    zhatvar1 = mean(zp(:,3)./zpsum - zhat1.^2);
    if isnan(zhat0)
        disp('wrong ratio0'+str(sum(isnan(zhat0))/len(zhat0)));
        disp('wrong ratio1'+str(sum(isnan(zhat1))/len(zhat1)));
        zhat0(isnan(zhat0)) = 0;
        zhat1(isnan(zhat1)) = 0;
    end
end