function model = vbssa_train(X, E, Ls, m, opts)
%   Inputs:
%     X    : N x M training data matrix (N samples, M observed variables)
%     E    : number of consecutive epochs
%     Ls   : number of stationary sources (SSs); L = M sources in total,
%            the remaining L-Ls are nonstationary sources (NSs)
%     m    : mixture order per source. Scalar or L x 1 vector.
%            m(j)=1 for all j  -> VbSSA-GC ; m(j)>1 -> VbSSA-HGC
%     opts : (optional) struct with fields
%            .maxIter   outer (second-hierarchy) iterations      [50]
%            .innerIter first-hierarchy s/A iterations           [10]
%            .tol       convergence tolerance on free energy     [1e-5]
%            .lambda,.a,.b,.alpha,.beta,.c,.d,.v : prior hyperparameters
%
%   Output:
%     model : struct containing all variational posterior parameters,



% --------------------------- preprocessing -------------------------------
[N, M] = size(X);
L = M;                       % square, invertible mixing matrix
Ln = L - Ls;                 % number of NSs
if nargin < 5, opts = struct; end
if ~isfield(opts,'maxIter'),   opts.maxIter   = 50;    end
if ~isfield(opts,'innerIter'), opts.innerIter = 10;    end
if ~isfield(opts,'tol'),       opts.tol       = 1e-5;  end
% prior hyperparameters {lambda_j, a_j, b_j, alpha_j, beta_j, c_k, d_k, v_kj}
if ~isfield(opts,'lambda'), opts.lambda = 1;    end
if ~isfield(opts,'a'),      opts.a      = 0;    end
if ~isfield(opts,'b'),      opts.b      = 1;    end
if ~isfield(opts,'alpha'),  opts.alpha  = 2;    end
if ~isfield(opts,'beta'),   opts.beta   = 1;    end
if ~isfield(opts,'c'),      opts.c      = 1e-3; end
if ~isfield(opts,'d'),      opts.d      = 1e-3; end
if ~isfield(opts,'v'),      opts.v      = 1e-2; end

if isscalar(m), m = m*ones(L,1); end
mmax = max(m);

Xraw = X;
muX  = mean(X,1);
sigX = std(X,0,1); sigX(sigX<eps) = 1;
X    = (X - muX)./sigX;

bnd  = round(linspace(0,N,E+1));
eidx = zeros(N,1);
for i = 1:E, eidx(bnd(i)+1:bnd(i+1)) = i; end

% --------------------------- initialization ------------------------------
rng(0);
mA = eye(M) + 0.05*randn(M,L);         
vA = 10*ones(M,L);                     
ED    = ones(M,1);                  
ElogD = zeros(M,1);                     
c_hat = opts.c*ones(M,1);
d_hat = opts.d*ones(M,1);

gamma = zeros(N,L,mmax);          
for j = 1:L, gamma(:,j,1:m(j)) = 1/m(j); end


aS = zeros(Ls,mmax);  bS = opts.b*ones(Ls,mmax);      
alS = opts.alpha*ones(Ls,mmax); beS = opts.beta*ones(Ls,mmax); 
laS = opts.lambda*ones(Ls,mmax);                      

aN = zeros(E,Ln,mmax);  bN = opts.b*ones(E,Ln,mmax);
alN = opts.alpha*ones(E,Ln,mmax); beN = opts.beta*ones(E,Ln,mmax);
laN = opts.lambda*ones(E,Ln,mmax);

% source posterior moments <s_j(t)> and Var(s_j(t))
muS  = zeros(N,L);
varS = ones(N,L);

s0 = X/mA.';
for j = 1:L
    sd = std(s0(:,j))+eps;
    if j <= Ls
        aS(j,1:m(j)) = linspace(-sd,sd,m(j));
    else
        for i = 1:E
            aN(i,j-Ls,1:m(j)) = linspace(mean(s0(eidx==i,j))-sd, ...
                                         mean(s0(eidx==i,j))+sd, m(j));
        end
    end
end

Fold = -inf;

% ============================ main VB loop ===============================
for iter = 1:opts.maxIter


    EA  = mA;
    EA2 = mA.^2 + 1./vA;                 % <A_kj^2>
    for inner = 1:opts.innerIter
        muS_old = muS;
        recon = muS*mA.';                % <x~_k(t)> = sum_j <A_kj><s_j(t)>
        for j = 1:L
            [ESig,ESigmu] = comp_expect(j,eidx,E,m,alS,beS,aS,alN,beN,aN,Ls);
            gam = gamma(:,j,1:m(j));     % N x m_j
            % Sigma^_j(t), Eq. (A.15), 2nd line
            Sig = gam*ESig.' + sum(ED.'.*EA2(:,j).');      % N x 1
            % x_k(t) - <x~_{k,k~=j}(t)>
            xres = X - recon + muS(:,j)*mA(:,j).';         % N x M
            coef = ED.*mA(:,j);                            % <Delta_k A_kj>
            % mu^_j(t), Eq. (A.15), 1st line
            muS(:,j)  = (gam*ESigmu.' + xres*coef)./Sig;
            varS(:,j) = 1./Sig;
            recon = recon + (muS(:,j)-muS_old(:,j))*mA(:,j).';
        end
        s2  = varS + muS.^2;           
        for j = 1:L
            xres = X - muS*mA.' + muS(:,j)*mA(:,j).';
            vA(:,j) = opts.v + ED.*sum(s2(:,j)).';         % v^_kj
            mA(:,j) = (ED.*(xres.'*muS(:,j)))./vA(:,j);    % m^_kj
        end
        EA  = mA; EA2 = mA.^2 + 1./vA;
        if max(abs(muS(:)-muS_old(:))) < opts.tol, break; end
    end


    for j = 1:L
        [ESig,~,ElogSig,logpi,aa,bb] = comp_all(j,eidx,E,m, ...
                       alS,beS,aS,bS,laS, alN,beN,aN,bN,laN, Ls);

        q2 = varS(:,j) + muS(:,j).^2 - 2*muS(:,j).*aa ...
             + aa.^2 + 1./bb;                             % N x m_j
        logg = logpi + 0.5*ElogSig - 0.5*ESig.*q2;
        logg = logg - max(logg,[],2);
        g = exp(logg);
        gamma(:,j,1:m(j)) = g./sum(g,2);                  % Eq. (A.11)
    end


    s2 = varS + muS.^2;
    for j = 1:L
        if j <= Ls                                        % SS: all samples
            gam = gamma(:,j,1:m(j));
            Nq  = sum(gam,1).';                           % sum_t gamma^
            [ESig,~,~,~,~,~] = comp_all(j,eidx,E,m,alS,beS,aS,bS,laS, ...
                                        alN,beN,aN,bN,laN,Ls);
            es = ESig(1,:).';                             % m_j x 1
            bnew = opts.b + es.*Nq;                       % b^_j,q (A.9)
            anew = (opts.b*opts.a + es.*sum(gam.*muS(:,j),1).')./bnew;
            alnew = opts.alpha + 0.5*Nq;                  % alpha^_j,q (A.10)
            % <(s-mu)^2> with posterior over mu:
            q2 = varS(:,j) + muS(:,j).^2 - 2*muS(:,j).*anew.' ...
                 + anew.'.^2 + 1./bnew.';
            benew = opts.beta + 0.5*sum(gam.*q2,1).';     % beta^_j,q
            bS(j,1:m(j))=bnew.'; aS(j,1:m(j))=anew.';
            alS(j,1:m(j))=alnew.'; beS(j,1:m(j))=benew.';
            laS(j,1:m(j))=opts.lambda + Nq.';             % lambda^_j,q (A.8)
        else                                              % NS: epoch-wise
            jn = j - Ls;
            for i = 1:E
                idx = (eidx==i);
                gam = gamma(idx,j,1:m(j));
                Nq  = sum(gam,1).';
                ESig = squeeze(alN(i,jn,1:m(j)))./squeeze(beN(i,jn,1:m(j)));
                bnew = opts.b + ESig.*Nq;
                anew = (opts.b*opts.a + ESig.*sum(gam.*muS(idx,j),1).')./bnew;
                alnew = opts.alpha + 0.5*Nq;
                q2 = varS(idx,j) + muS(idx,j).^2 - 2*muS(idx,j).*anew.' ...
                     + anew.'.^2 + 1./bnew.';
                benew = opts.beta + 0.5*sum(gam.*q2,1).';
                bN(i,jn,1:m(j))=bnew.';  aN(i,jn,1:m(j))=anew.';
                alN(i,jn,1:m(j))=alnew.'; beN(i,jn,1:m(j))=benew.';
                laN(i,jn,1:m(j))=opts.lambda + Nq.';
            end
        end
    end


    recon = muS*mA.';
    c_hat = opts.c + N/2;                                 % c^_k
    d_hat = opts.d + 0.5*sum((X-recon).^2,1).';           % d^_k
    ED    = c_hat./d_hat;                                 % <Delta_k>
    ElogD = psi(c_hat) - log(d_hat);


    F = vbssa_free_energy(X,eidx,E,m,gamma,muS,varS,mA,vA,ED,ElogD, ...
         aS,bS,alS,beS,laS,aN,bN,alN,beN,laN,Ls,opts,c_hat,d_hat);
    if abs(F-Fold) < opts.tol*max(1,abs(F)), break; end
    Fold = F;
end
fprintf('VbSSA training finished at iteration %d, free energy = %.4f\n',iter,F);


model = struct;
model.muX = muX; model.sigX = sigX; model.eidx = eidx; model.E = E;
model.L = L; model.Ls = Ls; model.m = m;
model.mA = mA; model.vA = vA;
model.c_hat = c_hat; model.d_hat = d_hat; model.ED = ED;
model.gamma = gamma; model.muS = muS; model.varS = varS;
model.aS=aS; model.bS=bS; model.alS=alS; model.beS=beS; model.laS=laS;
model.aN=aN; model.bN=bN; model.alN=alN; model.beN=beN; model.laN=laN;
model.free_energy = F;


model.mubar = zeros(E,L);      % mu^i  (collapsed component means)
model.Scol  = zeros(E,L);      % S_i diagonal (collapsed variances)
for j = 1:L
    if j <= Ls
        pi_ = laS(j,1:m(j))/sum(laS(j,1:m(j)));
        a_  = aS(j,1:m(j));
        sg2 = beS(j,1:m(j))./max(alS(j,1:m(j))-1,eps);   % sigma^^2, Eq.(22)
        muj = sum(pi_.*a_);
        Sj  = sum(pi_.*(sg2 + a_.^2)) - muj^2;
        model.mubar(:,j) = muj;
        model.Scol(:,j)  = Sj;
    else
        jn = j - Ls;
        for i = 1:E
            pi_ = squeeze(laN(i,jn,1:m(j))).';
            pi_ = pi_/sum(pi_);
            a_  = squeeze(aN(i,jn,1:m(j))).';
            sg2 = squeeze(beN(i,jn,1:m(j))).' ./ ...
                  max(squeeze(alN(i,jn,1:m(j))).'-1,eps);
            muj = sum(pi_.*a_);
            Sj  = sum(pi_.*(sg2 + a_.^2)) - muj^2;
            model.mubar(i,j) = muj;
            model.Scol(i,j)  = Sj;
        end
    end
end

model.trainingT2 = vbssa_monitor(model, Xraw);
end
function [ESig,ESigmu] = comp_expect(j,eidx,E,m,alS,beS,aS,alN,beN,aN,Ls)

N = numel(eidx);
if j <= Ls
    ESig   = repmat(alS(j,1:m(j))./beS(j,1:m(j)), N, 1);
    ESigmu = repmat(alS(j,1:m(j)).*aS(j,1:m(j))./beS(j,1:m(j)), N, 1);
else
    jn = j - Ls;
    ESig   = zeros(N,m(j)); ESigmu = zeros(N,m(j));
    for i = 1:E
        idx = (eidx==i);
        es  = squeeze(alN(i,jn,1:m(j))).'./squeeze(beN(i,jn,1:m(j))).';
        ESig(idx,:)   = repmat(es, nnz(idx), 1);
        ESigmu(idx,:) = repmat(es.*squeeze(aN(i,jn,1:m(j))).', nnz(idx), 1);
    end
end
end

function [ESig,ESigmu,ElogSig,logpi,aa,bb] = comp_all(j,eidx,E,m, ...
                                    alS,beS,aS,bS,laS, alN,beN,aN,bN,laN, Ls)
N = numel(eidx);
if j <= Ls
    es = alS(j,1:m(j))./beS(j,1:m(j));
    ESig    = repmat(es, N, 1);
    ESigmu  = repmat(es.*aS(j,1:m(j)), N, 1);
    ElogSig = repmat(psi(alS(j,1:m(j))) - log(beS(j,1:m(j))), N, 1); % (A.14)
    logpi   = repmat(psi(laS(j,1:m(j))) - psi(sum(laS(j,1:m(j)))), N, 1); %(A.12)
    aa      = repmat(aS(j,1:m(j)), N, 1);
    bb      = repmat(bS(j,1:m(j)), N, 1);
else
    jn = j - Ls;
    ESig=zeros(N,m(j)); ESigmu=zeros(N,m(j)); ElogSig=zeros(N,m(j));
    logpi=zeros(N,m(j)); aa=zeros(N,m(j)); bb=zeros(N,m(j));
    for i = 1:E
        idx = (eidx==i); n_ = nnz(idx);
        es = squeeze(alN(i,jn,1:m(j))).'./squeeze(beN(i,jn,1:m(j))).';
        ESig(idx,:)    = repmat(es, n_, 1);
        ESigmu(idx,:)  = repmat(es.*squeeze(aN(i,jn,1:m(j))).', n_, 1);
        ElogSig(idx,:) = repmat(psi(squeeze(alN(i,jn,1:m(j))).') - ...
                                log(squeeze(beN(i,jn,1:m(j))).'), n_, 1);
        l_ = squeeze(laN(i,jn,1:m(j))).';
        logpi(idx,:)   = repmat(psi(l_) - psi(sum(l_)), n_, 1);
        aa(idx,:)      = repmat(squeeze(aN(i,jn,1:m(j))).', n_, 1);
        bb(idx,:)      = repmat(squeeze(bN(i,jn,1:m(j))).', n_, 1);
    end
end
end


function F = vbssa_free_energy(X,eidx,E,m,gamma,muS,varS,mA,vA,ED,ElogD, ...
              aS,bS,alS,beS,laS,aN,bN,alN,beN,laN,Ls,opts,c_hat,d_hat)
[N,M] = size(X); L = M;
recon = muS*mA.';
res2 = zeros(N,M);
s2 = varS + muS.^2;
for k = 1:M
    r = X(:,k) - recon(:,k);
    % <|x_k - sum_j A_kj s_j|^2> keeping posterior variance terms
    r2 = r.^2 + s2*(mA(k,:).^2 + 1./vA(k,:)).' - (muS*mA(k,:).').^2;
    res2(:,k) = r2;
end
Fx = 0.5*sum(ElogD)*N - 0.5*sum(res2*ED) - 0.5*N*M*log(2*pi);


Fg = 0;
for j = 1:L
    gam = gamma(:,j,1:m(j));
    if j <= Ls
        es = alS(j,1:m(j))./beS(j,1:m(j));
        el = psi(alS(j,1:m(j))) - log(beS(j,1:m(j)));
        lp = psi(laS(j,1:m(j))) - psi(sum(laS(j,1:m(j))));
        q2 = varS(:,j) + muS(:,j).^2 - 2*muS(:,j)*aS(j,1:m(j)) ...
             + aS(j,1:m(j)).^2 + 1./bS(j,1:m(j));
        Fg = Fg + sum(sum(gam.*(repmat(lp+0.5*el,N,1) - 0.5*repmat(es,N,1).*q2)));
        Fg = Fg - sum(sum(gam.*log(gam+eps)));          % entropy of q
        % KL of component priors -> posteriors (Gamma & Gaussian & Dirichlet)
        Fg = Fg + gamma_kl(alS(j,1:m(j)),beS(j,1:m(j)),opts.alpha,opts.beta);
        Fg = Fg + gauss_kl(aS(j,1:m(j)),bS(j,1:m(j)),opts.a,opts.b);
        Fg = Fg + dir_kl(laS(j,1:m(j)),opts.lambda);
    else
        jn = j - Ls;
        for i = 1:E
            idx = (eidx==i);
            es = squeeze(alN(i,jn,1:m(j))).'./squeeze(beN(i,jn,1:m(j))).';
            el = psi(squeeze(alN(i,jn,1:m(j))).') - log(squeeze(beN(i,jn,1:m(j))).');
            l_ = squeeze(laN(i,jn,1:m(j))).';
            lp = psi(l_) - psi(sum(l_));
            a_ = squeeze(aN(i,jn,1:m(j))).'; b_ = squeeze(bN(i,jn,1:m(j))).';
            q2 = varS(idx,j) + muS(idx,j).^2 - 2*muS(idx,j).*a_ + a_.^2 + 1./b_;
            Fg = Fg + sum(sum(gam(idx,:).*(repmat(lp+0.5*el,nnz(idx),1) ...
                 - 0.5*repmat(es,nnz(idx),1).*q2)));
            Fg = Fg - sum(sum(gam(idx,:).*log(gam(idx,:)+eps)));
            Fg = Fg + gamma_kl(squeeze(alN(i,jn,1:m(j))).', ...
                               squeeze(beN(i,jn,1:m(j))).',opts.alpha,opts.beta);
            Fg = Fg + gauss_kl(a_,b_,opts.a,opts.b);
            Fg = Fg + dir_kl(l_,opts.lambda);
        end
    end
    Fg = Fg + 0.5*sum(log(2*pi*exp(1)*varS(:,j)));       % entropy of s_j
end

Fn = gamma_kl(c_hat.',d_hat.',opts.c,opts.d);
Fn = Fn + sum(gauss_kl(mA(:).',vA(:).',0,opts.v));

F = Fx + Fg + Fn;
end

function v = gamma_kl(a1,b1,a0,b0)   
v = sum( a0*log(b0) - gammaln(a0) - (a1*log(b1) - gammaln(a1)) ...
       + (a1-a0).*psi(a1) - (b1-b0).*a1./b1 );
end
function v = gauss_kl(a1,b1,a0,b0)   
% KL between scalar Gaussians parameterized by (mean, precision)
KL = 0.5*( log(b0./b1) + b1./b0 + b0.*(a1-a0).^2 - 1 );
v = -sum(KL);
end
function v = dir_kl(l1,l0)         
l0v = l0*ones(size(l1));
v = gammaln(sum(l0v)) - sum(gammaln(l0v)) ...
  - (gammaln(sum(l1)) - sum(gammaln(l1))) ...
  + sum((l1-l0v).*(psi(l1)-psi(sum(l1))));
end
