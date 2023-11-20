function [w, f, normal]=LevenbergMarquardt(r_and_J,w,kmax,tol)
f = zeros(kmax + 1,1);
normal = zeros(kmax,1);
Rmax=1;
R = 0.5; %Rmin=1e-14;
rho_good=0.75;
rho_bad = 0.25;
eta= 0.01;

for k = 1:kmax
    [r,J] = r_and_J(w);
    g=J'*r;
    normal(k) = norm(g);
    I = eye(size(J,2));
    B=J'*J + (1e-6)*I;
    pstar= -B\g; % unconstrained minimizer
    if norm(pstar) <=R
        p=pstar;
    else % solve constrained minimizer 
        lambda=1;
        while 1
            B1= B + lambda*I;
            C=chol(B1);
            p=-C\(C'\g);
            np =norm(p);
            dd= abs(np - R);
            if dd < 1e-6
                break
            end
            q=C'\p;
            nq=norm(q);
            lambda_new= lambda + (np/nq)^2*(np-R)/R;
            if lambda_new < 0
                lambda = 0.5 * lambda;
            else 
                lambda = lambda_new;
            end
        end
    end
    
    [r_xp,~] = r_and_J(w+p);
    m= 0.5*norm(r)^2+ p'*J'*r+ 0.5*p'*J'*J*p;
    f(k)=0.5*norm(r)^2;
    f_xp=0.5*norm(r_xp)^2;
    rho=(f(k)-f_xp)/(f(k)-m);
    if rho < rho_bad
        R=R*0.25;% reduce trust region
    else
        if rho> rho_good && norm(p)== R
            R=min(2*R,Rmax); % increase trust region
        end
    end
    if rho> eta
        w=w+p;%accept step
    else
        w=w; %reject step
    end
    if normal(k) < tol
        break;
    end

end
fprintf('k = %d, f = %d, ||g|| = %d\n',k,f(k),normal(k));
end
