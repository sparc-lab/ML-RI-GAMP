% State evolution (SE) recursion for ML-RI-GAMP

clear;
close all;
clc;

GAUSS = false;

delta1 = 2;
delta2 = 1.3;
sigmawgrid = 0.2; % noise standard deviation

niter = 28; % number of iterations of RI-GAMP
max_it = niter+2;

toleig = 10^(-10);

Nmc = 10^6; % number of MonteCarlo trials to compute integrals

% the first 60 rectangular free cumulants of the spectrum of the design
% matrix are pre-computed by and stored in mat files

if GAUSS
    freecum1 = zeros(1, 2*max_it);
    freecum1(1) = 1/delta1;
    freecum2 = zeros(1, 2*max_it);
    freecum2(1) = 1/delta2;
else
    if delta1==1
        load freerectcum_beta_60_delta1.mat;
    elseif delta1==0.5
        load freerectcum_beta_60_delta0_5.mat;
    elseif delta1==0.7
        load freerectcum_beta_60_delta0_7.mat;
    elseif delta1==1.1
        load freerectcum_beta_60_delta1_1.mat;
    elseif delta1==1.3
        load freerectcum_beta_60_delta1_3.mat;
    elseif delta1==1.7
        load freerectcum_beta_60_delta1_7.mat;
    elseif delta1==2
        load freerectcum_beta_60_delta2.mat;
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

tolinc = 10^(-6);
tolsmall = 10^(-3);
toldiff = 10^(-8);

% contains (limit) normalized squared correlations between the signal and 
% the x iterate produced by ML-RI-GAMP
scal_all1 = -ones(niter, length(sigmawgrid));

% contains (limit) normalized squared correlations between the signal and 
% the \hat{x} iterate produced by ML-RI-GAMP
scal_allhatx1 = -ones(niter, length(sigmawgrid));

% contains parameters \mu of SE
muall1 = zeros(niter, length(sigmawgrid));

% contains parameters \sigma of SE
sigmaall1 = zeros(niter, length(sigmawgrid));

% layer2
scal_all2 = -ones(niter, length(sigmawgrid));
scal_allhatx2 = -ones(niter, length(sigmawgrid));
muall2 = zeros(niter, length(sigmawgrid));
sigmaall2 = zeros(niter, length(sigmawgrid));


for ks = 1 : length(sigmawgrid)

    sigmaw = sigmawgrid(ks);
    
    fprintf('sigmaw = %f\n', sigmaw);

    % allocate vectors for SE recursion
    scal11 = zeros(niter, 1);
    muxSE1 = zeros(niter, 1);
    sigmaxSE1 = zeros(niter, 1);
    OmegaSE1 = zeros(niter+1, niter+1);
    SigmaSE1 = zeros(niter+1, niter+1);
    DeltaAUX1 = zeros(niter+1, niter+1);
    GammaAUX1 = zeros(niter+1, niter+1);
    PsiAUX1 = zeros(niter+1, niter+1);
    PhiAUX1 = zeros(niter+1, niter+1);

    % initialization of SE recursion (\Omega_1, \Sigma_1, \Delta_2,
    % \Gamma_1, \Phi_2)
    rho = 0.5;
    OmegaSE1(1, 1) = 0;
    if GAUSS
        DeltaAUX1(2, 2) = rho*freecum1(1)*(1+delta2)/delta2+sigmaw^2*delta2*freecum2(1);
    else
        DeltaAUX1(2, 2) = rho*freecum1(1)*6^2/((2+1)*(2*2+1))+sigmaw^2*delta2*freecum2(1);
    end
    GammaAUX1(1, 1) = 1;
    PhiAUX1(2, 1) = rho * delta2 * freecum2(1);
    
    % allocate vectors for SE recursion
    scal2 = zeros(niter, 1);
    muxSE2 = zeros(niter, 1);
    sigmaxSE2 = zeros(niter, 1);
    OmegaSE2 = zeros(niter+1, niter+1);
    SigmaSE2 = zeros(niter+1, niter+1);
    DeltaAUX2 = zeros(niter+1, niter+1);
    GammaAUX2 = zeros(niter+1, niter+1);
    PsiAUX2 = zeros(niter+1, niter+1);
    PhiAUX2 = zeros(niter+1, niter+1);
    auxMMall2 = zeros(niter, niter+1);
    auxMM2all2 = zeros(niter, niter);

    % initialization of SE recursion layer 2
    OmegaSE2(1, 1) = 0;
    GammaAUX2(1, 1) = rho*freecum1(1);
    SigmaSE2(1, 1) = freecum2(1)*GammaAUX2(1, 1);
    DeltaAUX2(2, 2) = SigmaSE2(1, 1) + sigmaw^2;
    PhiAUX2(2, 1) = 1;

    for t = 1 : niter
        %% layer1
        disp('layer1')
        % computation of \mu_t
        DeltaAUXtmp1 = DeltaAUX1(1:t+1, 1:t+1);
        GammaAUXtmp1 = GammaAUX1(1:t+1, 1:t+1);
        PsiAUXtmp1 = PsiAUX1(1:t+1, 1:t+1);
        PhiAUXtmp1 = PhiAUX1(1:t+1, 1:t+1);
        
        S1 = PsiAUXtmp1 * PhiAUXtmp1;

        muxSE1(t) = 0;

        for jj = 0 : t
            Mattmp = PhiAUXtmp1 * S1^jj;
            muxSE1(t) = muxSE1(t) + delta1 * freecum1(jj+1) * Mattmp(t+1, 1);
        end
        
        muall1(t, ks) = muxSE1(t);

        % computation of \Omega_{t+1}
        Q = PhiAUXtmp1 * PsiAUXtmp1;
        OmegaSEtmp1 = zeros(t+1, t+1);

        for t1 = 0 : 2*t
            pow = zeros(t+1, t+1);
            for t2 = 0 : t1
                pow = pow + Q^t2 * DeltaAUXtmp1 * (Q')^(t1-t2);
            end
            pow2 = zeros(t+1, t+1);
            for t2 = 0 : t1-1
                pow2 = pow2 + Q^t2 * PhiAUXtmp1 * GammaAUXtmp1 * PhiAUXtmp1' * (Q')^(t1-t2-1);
            end
            OmegaSEtmp1 = OmegaSEtmp1 + freecum1(t1+1) * (pow + pow2);
        end

        OmegaSEtmp1 = delta1 * OmegaSEtmp1;
        
        % \Omega_t must be a sub-matrix of \Omega_{t+1}        
        for j1 = 1 : t
            for j2 = 1 : t
                if abs(OmegaSEtmp1(j1, j2) - OmegaSE1(j1, j2)) > toldiff
                    Diff = OmegaSEtmp1(j1, j2)-OmegaSE1(j1, j2)
                    fprintf('Something is wrong with Omega(%d, %d) at iteration %d!\n Diff=%f\n', j1, j2, t, OmegaSEtmp1(j1, j2)-OmegaSE1(j1, j2));
                end
            end
        end

        OmegaSE1(1:t+1, 1:t+1) = OmegaSEtmp1;
        sigmaxSE1(t) = OmegaSE1(t+1, t+1);
        sigmaall1(t, ks) = sigmaxSE1(t);


        % this forces Omegared to be symmetric (it may not exactly be
        % because of numerical issues)    
        Omegaredtmp1 = OmegaSE1(2:t+1, 2:t+1);
        Omegared1 = 1/2 * ( Omegaredtmp1 + Omegaredtmp1' );

        % normalized squared correlation between x^t and the signal
        scal11(t) = muxSE1(t)/sqrt(muxSE1(t)^2+sigmaxSE1(t));
        scal_all1(t, ks) = scal11(t)^2;
        fprintf('Iteration %d, scal = %f\n', t, scal11(t)^2);

        % the matrix Omegared should be PSD
        if min(eig(Omegared1)) < toleig 
                fprintf('min(eig(Omega)) too small: Numerical precision exceeded! Warning!\n');
                min(eig(Omegared1))
                break;
        end

        % computation of \Psi_{t+1}
        PsiAUX1(t+1, 2:t+1) = muxSE1(1:t)' / (Omegared1 + muxSE1(1:t) * muxSE1(1:t)');    

        % computation of \Gamma_{t+1} (2 values)
        GammaAUX1(1, t+1) = muxSE1(1:t)' / (Omegared1 + muxSE1(1:t) * muxSE1(1:t)') * muxSE1(1:t);
        GammaAUX1(t+1, 1) = GammaAUX1(1, t+1);

        % normalized squared correlation between \hat{x}^t and the signal
        scal_allhatx1(t, ks) = GammaAUX1(t+1, 1);
        fprintf('hat-x, scal = %f\n', scal_allhatx1(t, ks));

        % computation of \Gamma_{t+1} (remaining part)
        for jj = 1 : t
            GammaAUX1(jj+1, t+1) = muxSE1(1:t)' / (Omegared1 + muxSE1(1:t) * muxSE1(1:t)') ...
                * (Omegared1(1:t, 1:jj) + muxSE1(1:t) * muxSE1(1:jj)')* ...
                (muxSE1(1:jj)' / (Omegared1(1:jj, 1:jj) + muxSE1(1:jj) * muxSE1(1:jj)'))';
            GammaAUX1(t+1, jj+1) = GammaAUX1(jj+1, t+1);
        end

        GammaAUXtmp1 = GammaAUX1(1:t+1, 1:t+1); 

        % computation of \Sigma_{t+1}
        PsiAUXtmp1 = PsiAUX1(1:t+1, 1:t+1);

        S1 = PsiAUXtmp1 * PhiAUXtmp1;
      
        SigmaSEtmp1 = zeros(t+1, t+1);
        for t1 = 0 : 2*t+1
            pow = zeros(t+1, t+1);
            for t2 = 0 : t1
                pow = pow + S1^t2 * GammaAUXtmp1 * (S1')^(t1-t2);
            end
            pow2 = zeros(t+1, t+1);
            for t2 = 0 : t1-1
                pow2 = pow2 + S1^t2 * PsiAUXtmp1 * DeltaAUXtmp1 * PsiAUXtmp1' * (S1')^(t1-t2-1);
            end
            SigmaSEtmp1 = SigmaSEtmp1 + freecum1(t1+1) * (pow + pow2);            
        end

        % \Sigma_t must be a sub-matrix of \Sigma_{t+1}
        for j1 = 1 : t
            for j2 = 1 : t
                if t>1 && abs(SigmaSEtmp1(j1, j2) - SigmaSE1(j1, j2)) > toldiff
                    Diff = SigmaSEtmp1(j1, j2)-SigmaSE1(j1, j2)
                    fprintf('Something is wrong with Sigma(%d, %d) at iteration %d!\n Diff=%f\n', j1, j2, t, SigmaSEtmp1(j1, j2)-SigmaSE1(j1, j2));
                end
            end
        end

        SigmaSE1(1:t+1, 1:t+1) = SigmaSEtmp1; 
        
        % computation of \Phi_{t+2}
        % the matrix Sigma(2:t+1, 2:t+1) should be PSD
        if ( min(eig(SigmaSE1(2:t+1, 2:t+1))) < toleig || min(eig(SigmaSE1(1:t+1, 1:t+1))) < toleig )
            fprintf('min(eig(Sigma)) too small: Numerical precision exceeded! Warning!\n');
            min(eig(SigmaSE1(1:t+1, 1:t+1)))
            min(eig(SigmaSE1(2:t+1, 2:t+1)))
            break;
        end
        
        % this forces Sigmared to be symmetric (it may not exactly be
        % because of numerical issues)    
        Sigmared1 = 1/2 * ( SigmaSEtmp1 + SigmaSEtmp1' );
        
        % for denoising of relu
        Rvec1 = (mvnrnd(zeros(1, t+1),Sigmared1,Nmc));
        gMC1 = Rvec1(:, 1)';
        rhatMC1 = SigmaSE1(1, 2:t+1) / SigmaSE1(2:t+1, 2:t+1) * Rvec1(:, 2:end)';
        sigma2hat1 = SigmaSE1(1, 1) - SigmaSE1(1, 2:t+1) / SigmaSE1(2:t+1, 2:t+1) * SigmaSE1(2:t+1, 1);
    %% layer2
        disp('layer 2')
        % computation of \mu_t
        DeltaAUXtmp2 = DeltaAUX2(1:t+1, 1:t+1);
        GammaAUXtmp2 = GammaAUX2(1:t+1, 1:t+1);
        PsiAUXtmp2 = PsiAUX2(1:t+1, 1:t+1);
        PhiAUXtmp2 = PhiAUX2(1:t+1, 1:t+1);

        S2 = PsiAUXtmp2 * PhiAUXtmp2;

        muxSE2(t) = 0;
        for jj = 0 : t
            Mattmp = PhiAUXtmp2 * S2^jj;
            muxSE2(t) = muxSE2(t) + delta2 * freecum2(jj+1) * Mattmp(t+1, 1);
        end
        
        muall2(t, ks) = muxSE2(t);

        % computation of \Omega_{t+1}
        Q = PhiAUXtmp2 * PsiAUXtmp2;
        OmegaSEtmp2 = zeros(t+1, t+1);

        for t1 = 0 : 2*t
            pow = zeros(t+1, t+1);
            for t2 = 0 : t1
                pow = pow + Q^t2 * DeltaAUXtmp2 * (Q')^(t1-t2);
            end
            pow2 = zeros(t+1, t+1);
            for t2 = 0 : t1-1
                pow2 = pow2 + Q^t2 * PhiAUXtmp2 * GammaAUXtmp2 * PhiAUXtmp2' * (Q')^(t1-t2-1);
            end
            OmegaSEtmp2 = OmegaSEtmp2 + freecum2(t1+1) * (pow + pow2);
        end

        OmegaSEtmp2 = delta2 * OmegaSEtmp2;

        % \Omega_t must be a sub-matrix of \Omega_{t+1}
        for j1 = 1 : t
            for j2 = 1 : t
                if t>1 && abs(OmegaSEtmp2(j1, j2) - OmegaSE2(j1, j2)) > toldiff
                    Diff = OmegaSEtmp2(j1, j2)-OmegaSE2(j1, j2)
                    fprintf('Something is wrong with Omega(%d, %d) at iteration %d!\n Diff=%f\n', j1, j2, t, OmegaSEtmp2(j1, j2)-OmegaSE2(j1, j2));
                end
            end
        end

        OmegaSE2(1:t+1, 1:t+1) = OmegaSEtmp2;
        sigmaxSE2(t) = OmegaSE2(t+1, t+1);
        
        sigmaall2(t, ks) = sigmaxSE2(t);

        %   this forces Omegared to be symmetric (it may not exactly be
        % because of numerical issues)    
        Omegaredtmp2 = OmegaSE2(2:t+1, 2:t+1);
        Omegared2 = 1/2 * ( Omegaredtmp2 + Omegaredtmp2' );

        % normalized squared correlation between x^t and the signal
        scal2(t) = muxSE2(t)*sqrt(rho)/sqrt(muxSE2(t)^2*rho+sigmaxSE2(t));
        scal_all2(t, ks) = scal2(t)^2;
        fprintf('Iteration %d, scal = %f\n', t, scal2(t)^2);
        
        % the matrix Omegared should be PSD
        if min(eig(Omegared2)) < toleig 
                fprintf('min(eig(Omega)) too small: Numerical precision exceeded! Warning!\n');
                min(eig(Omegared2))
                break;
        end

        % computation of \Psi_{t+1}
        Wvec2 = (mvnrnd(zeros(1, t),Omegared2,Nmc))';
        vpd2 = muxSE2(1:t)' / Omegared2;
        
        % denoising of relu
        xtrueMC2 = max(gMC1,0);
        vpd2 = muxSE2(1:t)' / Omegared2(1:t, 1:t);
        xMC2 = muxSE2(1:t) * xtrueMC2 + Wvec2;
        rhot2 = 1/(vpd2 * muxSE2(1:t));
        xtMC2 = vpd2 * xMC2 * rhot2;

        [zhat0,zhat1,zhatvar0,zhatvar1] = est_mmse(real(rhatMC1)',real(xtMC2)',real(sigma2hat1),rhot2);
        xhatMC2 = zhat1';            
        PsiAUX2(t+1, 2:t+1) = zhatvar1/rhot2 * muxSE2(1:t)'/Omegared2*rhot2;   

        % computation of \Gamma_{t+1} (2 values)
        GammaAUX2(1, t+1) = mean(xhatMC2.*xtrueMC2);
        GammaAUX2(t+1, 1) = GammaAUX2(1, t+1);

        % normalized squared correlation between \hat{x}^t and the signal
        scal_allhatx2(t, ks) = GammaAUX2(t+1, 1)^2/mean(xhatMC2.^2)/mean(xtrueMC2.^2);
        fprintf('hat-x, scal = %f\n', scal_allhatx2(t, ks));

        % computation of \Gamma_{t+1} (remaining part)
        for jj = 1 : t
            muR2 = muxSE2(1:jj);
            vpdR2 = muR2' / Omegared2(1:jj, 1:jj);
            WvecR2 = Wvec2(1:jj, :);
            xMCR2 = muR2 * xtrueMC2 + WvecR2;
            rhotR2 = 1/(vpdR2 * muR2);
            xtMCR2 = vpdR2 * xMCR2 * rhotR2;

            rhatMCjj1 = SigmaSE1(1, 2:jj+1) / SigmaSE1(2:jj+1, 2:jj+1) * Rvec1(:, 2:jj+1)';
            sigma2hatjj1 = SigmaSE1(1, 1) - SigmaSE1(1, 2:jj+1) / SigmaSE1(2:jj+1, 2:jj+1) * SigmaSE1(2:jj+1, 1);

            [~,zhatR1,~,~] = est_mmse(real(rhatMCjj1)',real(xtMCR2)',real(sigma2hatjj1),rhotR2);
            xhatMCR2 = zhatR1';

            GammaAUX2(t+1, jj+1) = mean(xhatMC2.*xhatMCR2);
            GammaAUX2(jj+1, t+1) = GammaAUX2(t+1, jj+1);
        end

        GammaAUXtmp2 = GammaAUX2(1:t+1, 1:t+1);
        
        % computation of \Sigma_{t+1}
        PsiAUXtmp2 = PsiAUX2(1:t+1, 1:t+1);
        %% layer1 s
        sMC1 = zhat0' - rhatMC1;
        PhiAUX1(t+2, 1) = mean(sMC1.^2)/sigma2hat1;
        PhiAUX1(t+2, 2:t+1) = (zhatvar0/sigma2hat1-1) * SigmaSE1(1, 2:t+1) / SigmaSE1(2:t+1, 2:t+1);
        muR2 = muxSE2(1);
        vpdR2 = muR2' / Omegared2(1, 1);
        WvecR2 = Wvec2(1, :);
        xMCR2 = muR2 * xtrueMC2 + WvecR2;
                
        s1 = xMCR2;
        DeltaAUX1(2, t+2) = mean(sMC1.* s1);
        DeltaAUX1(t+2, 2) = DeltaAUX1(2, t+2);

        for jj = 2 : t+1  
            rhatMCjj1 = SigmaSE1(1, 2:jj) / SigmaSE1(2:jj, 2:jj) * Rvec1(:, 2:jj)';
            sigma2hatjj1 = SigmaSE1(1, 1) - SigmaSE1(1, 2:jj) / SigmaSE1(2:jj, 2:jj) * SigmaSE1(2:jj, 1);

            muR2 = muxSE2(1:jj-1);
            vpdR2 = muR2' / Omegared2(1:jj-1, 1:jj-1);
            WvecR2 = Wvec2(1:jj-1, :);
            xMCR2 = muR2 * xtrueMC2 + WvecR2;
            rhotR2 = 1/(vpdR2 * muR2);
            xtMCR2 = vpdR2 * xMCR2 * rhotR2;

            [zhatjj0,~,~,~] = est_mmse(real(rhatMCjj1)',real(xtMCR2)',real(sigma2hatjj1),real(rhotR2));

            sMCjj1 = zhatjj0' - rhatMCjj1;
            DeltaAUX1(jj+1, t+2) = mean(sMC1.*sMCjj1);    
            DeltaAUX1(t+2, jj+1) = DeltaAUX1(jj+1, t+2);
        end
        
        %% layer2, s
        S2 = PsiAUXtmp2 * PhiAUXtmp2;
        SigmaSEtmp2 = zeros(t+1, t+1);
        for t1 = 0 : 2*t+1
            pow = zeros(t+1, t+1);
            for t2 = 0 : t1
                pow = pow + S2^t2 * GammaAUXtmp2 * (S2')^(t1-t2);
            end           
            pow2 = zeros(t+1, t+1);
            for t2 = 0 : t1-1
                pow2 = pow2 + S2^t2 * PsiAUXtmp2 * DeltaAUXtmp2 * PsiAUXtmp2' * (S2')^(t1-t2-1);
            end
            SigmaSEtmp2 = SigmaSEtmp2 + freecum2(t1+1) * (pow + pow2);
        end

        % \Sigma_t must be a sub-matrix of \Sigma_{t+1}
        for j1 = 1 : t
            for j2 = 1 : t
                if abs(SigmaSEtmp2(j1, j2) - SigmaSE2(j1, j2)) > toldiff
                    Diff = SigmaSEtmp2(j1, j2)-SigmaSE2(j1, j2)
                    fprintf('Something is wrong with Sigma(%d, %d) at iteration %d!\n Diff=%f\n', j1, j2, t, SigmaSEtmp2(j1, j2)-SigmaSE2(j1, j2));
                end
            end
        end

        SigmaSE2(1:t+1, 1:t+1) = SigmaSEtmp2;

        MM = [SigmaSE2(1:t+1, 1:t+1), SigmaSE2(1:t+1, 1); SigmaSE2(1, 1:t+1), SigmaSE2(1, 1) + sigmaw^2];

        % computation of \Phi_{t+2}
        
        % the matrix MM(2:t+2, 2:t+2) should be PSD
        if min(eig((MM(2:end, 2:end)))) < toleig 
            fprintf('min(eig(MM)) too small: Numerical precision exceeded! Warning!\n');  
            min(eig((MM(2:end, 2:end))))
        end

        auxMM = MM(1, 2:end) / MM(2:end, 2:end);
        auxMMall2(t, 1:t+1) = auxMM;

        % the matrix Sigma(2:t+1, 2:t+1) should be PSD
        if min(eig(SigmaSE2(2:t+1, 2:t+1))) < toleig 
            fprintf('min(eig(Sigma)) too small: Numerical precision exceeded! Warning!\n');
            min(eig(SigmaSE2(2:t+1, 2:t+1)))
            break;
        end

        auxMM2 = SigmaSE2(1, 2:t+1) / SigmaSE2(2:t+1, 2:t+1);
        auxMM2all2(t, 1:t) = auxMM2;
        PhiAUX2(t+2, 1) = auxMM(t+1);

        for j2 = 1 : t
            PhiAUX2(t+2, j2+1) = auxMM(j2)-auxMM2(j2);    
        end

        % computation of \Delta_{t+2}
        DeltaAUX2(2, t+2) = auxMM * MM(2:t+2, t+2) - auxMM2 * SigmaSE2(2:t+1, 1);
        DeltaAUX2(t+2, 2) = DeltaAUX2(2, t+2);    

        for jj = 2 : t+1
            DeltaAUX2(jj+1, t+2) = auxMM * MM(2:t+2, [2:jj, t+2]) * auxMMall2(jj-1, 1:jj)' ...
                + auxMM2 * MM(2:t+1, 2:jj) * auxMM2all2(jj-1, 1:jj-1)' ...
                - auxMM2 * MM(2:t+1, [2:jj, t+2]) * auxMMall2(jj-1, 1:jj)' ...
                - auxMM * MM(2:t+2, 2:jj) * auxMM2all2(jj-1, 1:jj-1)'; ...
            DeltaAUX2(t+2, jj+1) = DeltaAUX2(jj+1, t+2);
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