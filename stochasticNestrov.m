function [w, f, normgradE] = stochasticNestrov(bsz,fun,gfun,w,kmax,tol)
    normgrad = zeros(kmax,1);
    normgradE = zeros(15,1);
    f = zeros(kmax + 1,1);
    n=13007;
    epoch =1;
    k = 1;
    normgrad(k) = 1;
    lowestNG =100;
    w0=w;
    while epoch < 15
    Ig=randperm(n);%generate a random permutation of 1 to n
    b=1;
    while b<n % perform SG on a batch worth of indexes
        w=w0;
        w_old=w;
        alp=0.3;
        for k=1:kmax
            Ig_i = Ig(b:min(b+bsz,n)); % Take a vector out of the batch
            mu=1-3/(5+k);
            y=(1+mu)*w - mu*w_old; %gradient update
            g_y=gfun(Ig_i,y); % estimate the gradient based on the vector
            w_old=w;
            w=y-alp*g_y;
            f(k)=fun(Ig_i,w); % estimate the value of the function
            g=gfun(Ig_i,w); % estimate the gradient based on the vector
            normgrad(k)=norm(g); % norm of the estimated gradient
            if normgrad(k) < tol
                break;
            end
        end
        b=b+bsz;
    end
    if normgrad(k)< lowestNG
        bestW=w;
    end
    normgradE(epoch)=norm(g);
    fprintf('epoch: %d, k = %d, f = %d, ||g|| = %d\n',epoch,k,f(k),normgrad(k));
    epoch = epoch +1;
    end
    w= bestW;
    f=fun(Ig,w);
end
