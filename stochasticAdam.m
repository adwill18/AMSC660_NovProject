function [w, f, normgradE] = stochasticAdam(bsz,fun,gfun,w,kmax,tol)
    beta1=0.9;
    beta2=0.999;
    eps=10^-8;
    eta=0.001;
    normgrad = zeros(kmax,1);
    normgradE = zeros(15,1);
    f = zeros(kmax + 1,1);
    n=13007;
    epoch =1;
    k = 1;
    lowestNG=100;
    normgrad(k) = 1;
    w0=w;
    while epoch < 15
        Ig=randperm(n);%generate a random permutation of 1 to n
        b=1;
        w=w0;
        while b<n % perform SG on a batch worth of indexes
            v=0;
            m=0;
            for k=1:kmax
                Ig_i = Ig(b:min(b+bsz,n)); % Take a vector out of the batch
                g=gfun(Ig_i,w); % estimate the gradient based on the vector
                f(k)=fun(Ig_i,w); % estimate the value of the function
                normgrad(k)=norm(g); % norm of the estimated gradient
                m=beta1*m+(1-beta1)*g;
                v=beta2*v+(1-beta2)*(g.*g);
                m_hat=m/(1-beta1);
                v_hat=v/(1-beta2);
                w=w-eta*m_hat./(sqrt(v_hat)+eps);
                if normgrad(k) < tol
                    break;
                end
            end
            b=b+bsz;
        end
        if normgrad(k)< lowestNG
            bestW=w;
            bestF=f;
        end
        normgradE(epoch)=norm(g);
        fprintf('epoch: %d, k = %d, f = %d, ||g|| = %d\n',epoch,k,f(k),normgrad(k));
        epoch = epoch +1;
    end
    w= bestW;
    f=bestF;
end
