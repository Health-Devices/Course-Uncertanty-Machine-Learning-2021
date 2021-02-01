using Turing



@model school8_mv(J, y, sigma, gen=false) = begin
    mu ~ Normal(0, 5)
    tau ~ Truncated(Cauchy(0, 5), 0, Inf)
    #eta = tzeros(J)
    eta ~ MvNormal(fill(0.0,J), fill(1.0,J))
    theta = mu .+ tau .* eta
    if gen == true
        y_hat = Vector{Real}(undef, J)
        log_lik = Vector{Real}(undef, J)
        for j = 1:J
            dist = Normal(theta[j], sigma[j])
            y[j] ~ dist
            y_hat[j] = rand(dist)
            log_lik[j] = logpdf(dist, y[j])
        end
        return (mu=mu, tau=tau, eta=eta, theta=theta, y_hat=y_hat, log_lik=log_lik)
    end
end

J = 8;
y = [28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0];
sigma = [15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0];


model_fun_mv = school8_mv(J, y, sigma, true);
#model_fun() # correctly draws a NamedTuple


chns_mv = sample(model_fun_mv, NUTS(2000,0.8), 4000) # lacks theta, y_hat, and log_lik


mixeddensity!(chns_mv,[:mu, :tau])

function get_nlogp(model,sm)
    # Set up the model call, sample from the prior.
    vi = Turing.VarInfo(model)
    # Define a function to optimize.
    function nlogp(sm)
        spl = Turing.SampleFromPrior()
        new_vi = Turing.VarInfo(vi, spl, sm)
        a=model(new_vi, spl)
        return(a)
    end
    return nlogp(sm)
end


model_fun_mv_ppc = school8_mv(J, y, sigma, true)

mu_post_mv = Array(chns_mv[:mu])
tau_post_mv = Array(chns_mv[:tau])
eta_post_mv = (get(chns_mv,:eta))

get_nlogp(model_fun_mv_ppc,vcat(mu_post_mv[1],tau_post_mv[1],eta_post_mv.eta[1][1,:]))
