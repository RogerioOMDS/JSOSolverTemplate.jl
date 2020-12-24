using LinearAlgebra, ForwardDiff, NLPModels, Random
Random.seed!(0)

export LBFGS_StrongWolfe

include("StrongWolfe.jl")

function LBFGS_StrongWolfe(
  nlp::AbstractNLPModel;
  m = 3,  # Memory reserved
  a1 = 1.0, # Initial Step
  η1::Float64 = 1e-4, # Armijo Condition
  η2::Float64 = 0.8, # StrongWolfe
  p::Float64 = 0.9, # Line Search Step
  atol::Real = 1e-6,
  rtol::Real = 1e-6,
  max_eval::Int = 50000,
  max_iter::Int = 0,
  max_time::Float64 = 60.0,
  )
  
  if !unconstrained(nlp)
    error("Problem is not unconstrained")
  end
  
  x = copy(nlp.meta.x0)
  n = nlp.meta.nvar
  
  f(x) = obj(nlp, x)
  ∇f(x) = grad(nlp, x)
  fx = f(x)
  ∇fx = ∇f(x)
  
  ϵ = atol + rtol * norm(∇f(x))
  t₀ = time()
  
  iter = 0
  Δt = time() - t₀
  solved = norm(∇fx) < ϵ # First order stationary
  tired = neval_obj(nlp) ≥ max_eval > 0|| iter ≥ max_iter > 0 || Δt ≥ max_time > 0 # Excess time, iteration, evaluations
  
  status = :unknown
  
  @info log_header(
  [:iter, :fx, :ngx, :nf, :Δt],
  [Int, Float64, Float64, Int, Float64],
  hdr_override=Dict(:fx => "f(x)", :ngx => "‖∇f(x)‖", :nf => "#f")
  )
  @info log_row(
  Any[iter, fx, norm(∇fx), neval_obj(nlp), Δt]
  )
  
  a = zeros(m)
  y = zeros(n ,m)
  s = zeros(n, m)
  Id = Matrix(1.0I, n, n)
  
  r = - ∇fx

  α, iteracao = StrongWolfe_LineSearch(nlp, x, r, a1, η1, η2, p) #Calcular um passo com α de Wolfe
  xt = x + α * r
  s[:, 1] = xt - x  
  y[:, 1] = ∇f(xt) - ∇fx
  x = xt
  # γ = dot(s[:, 1], y[:, 1]) / dot(y[:, 1], y[:, 1]) 
  # H0 = γ * Id
  
  while !(solved || tired)
    γ = dot(s[:, iter % m + 1], y[:, iter % m + 1]) / dot(y[:, iter % m + 1], y[:, iter % m + 1])
    H0 = γ * Id  

    if iter < m
      r = - H0 * ∇fx
    else
      q = ∇fx
      for i = ((iter - 1) % m + 1):-1:((iter - m) % m + 1)
        ρ = 1 / dot(y[:,i],s[:,i])
        a[i] = ρ * dot(s[:,i], q)
        q -= a[i] * y[:,i]
      end
      r = H0 * q
      for i = ((iter - m) % m + 1):((iter - 1) % m + 1)
        ρ = 1 / dot(y[:,i],s[:,i])
        β = ρ * dot(y[:,i],r)
        r += s[:,i] * (a[i] - β) # retorna a direção r = Hₖ∇f
        r = -r
      end
    end

    α, iteracao = StrongWolfe_LineSearch(nlp, x, r, a1, η1, η2, p) #Calcular um passo com α de Wolfe
    xt = x + α * r #Atualiza o passo xₖ₊₁ = xₖ - αₖ rₖ  

    #calcular sₖ e yₖ para próxima iteração
    s[:, (iter + 1) % m + 1] = xt - x   
    y[:, (iter + 1) % m + 1] = ∇f(xt) - ∇fx
    x = xt

    fx = f(x)
    ∇fx = ∇f(x)

    iter += 1
    
    Δt = time() - t₀
    solved = norm(∇fx) < ϵ # First order stationary
    tired = neval_obj(nlp) ≥ max_eval > 0 || iter ≥ max_iter > 0 || Δt ≥ max_time > 0 # Excess time, iteration, evaluations
    
    @info log_row(
    Any[iter, fx, norm(∇fx), neval_obj(nlp), Δt]
    )
  end
  
  if solved
    status = :first_order
  elseif tired
    if neval_obj(nlp) ≥ max_eval > 0
      status = :max_eval
    elseif iter ≥ max_iter > 0
      status = :max_iter
    elseif Δt ≥ max_time > 0
      status = :max_time
    end
  end
  
  return GenericExecutionStats(
  status,
  nlp,
  solution = x,
  objective = f(x),
  dual_feas = norm(∇f(x)),
  elapsed_time = Δt,
  iter = iter
  )
end
