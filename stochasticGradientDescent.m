function [w, f, normgradE] = stochasticGradientDescent(bsz,fun,gfun,w,kmax,tol)
    normgrad = zeros(kmax,1);
    normgradE = zeros(15,1);
    f = zeros(kmax + 1,1);
    
    n=13007;
    k = 1;
    kk=1;
    normgrad(k) = 1;
    epoch =1;
    lowestNG=100;
    normgrad(k) = 1;
    w0=w;
    while epoch < 15
    Ig=randperm(n);%generate a random permutation of 1 to n
    b=1;
    w=w0;
    while b<n % perform SG on a batch worth of indexes
        step0=0.3;
        for k=1:kmax
            Ig_i = Ig(b:min(b+bsz,n)); % Take a vector out of the batch
            g=gfun(Ig_i,w); % estimate the gradient based on the vector
            f(k)=fun(Ig_i,w); % estimate the value of the function
            normgrad(k)=norm(g); % norm of the estimated gradient
            alp=step0/(k^3);
            w=w-alp*g;
            if normgrad(k) < tol
                break;
            end 
            kk=kk+1;
        end
    b=b+bsz;
    end
    if normgrad(k)< lowestNG
       bestW=w;
       bestF=f;
    end
    normgradE(epoch)=norm(g);
    fprintf('epoch: %d, k = %d, k_tot= %d, f = %d, ||g|| = %d\n',epoch,k,kk,f(k),normgrad(k));
    epoch = epoch +1;
    end
    w= bestW;
