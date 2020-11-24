using DifferentialEquations
using BenchmarkTools
using NCDatasets


function double_pendulum!(du, u, p, t)
    θ₁, θ₂, dθ₁, dθ₂ = u
    g = 9.81
    L₁, L₂, m₁, m₂ = p
    A₁ = -g * (2m₁ + m₂) * sin(θ₁) - m₂ * g * sin(θ₁-2θ₂) - 2sin(θ₁ - θ₂) * m₂*(dθ₂^2 * L₂ + dθ₁^2 * L₁ * cos(θ₁ - θ₂))
    A₂ = 2sin(θ₁ - θ₂) * (dθ₁^2 * L₁ * (m₁ + m₂) + g * (m₁ + m₂) * cos(θ₁) + dθ₂^2 * L₂ * m₂ * cos(θ₁ - θ₂))
    B = 2m₁ + m₂ - m₂ * cos(2θ₁ - 2θ₂)
    B₁ = L₁ * B
    B₂ = L₂ * B
    du[1] = dθ₁
    du[2] = dθ₂
    du[3] = ddθ₁ = A₁ / B₁
    du[4] = ddθ₂ = A₂ / B₂
end

L₁ = 0.1
L₂ = 0.1
m₁ = 0.05
m₂ = 0.05
p = [L₁, L₂, m₁, m₂]

# θ₁₀ = 2.
# θ₂₀ = 2.
# dθ₁₀ = 0.
# dθ₂₀ = 0.
# u₀ = [θ₁₀, θ₂₀, dθ₁₀, dθ₂₀]

@info Threads.nthreads()

tspan = (0., 100.)

# prob = ODEProblem(double_pendulum!, u₀, tspan, p, saveat = 0.1)

# sol₁ = solve(prob, abstol=1e-15, reltol=1e-15)
# sol₂ = solve(prob, abstol=1e-14, reltol=1e-14)

function initialise_u₀(range₁, range₂)
    initial_conditions = Array{Array{Float64, 1}}(undef, length(range₁), length(range₂))
    for j in 1:length(range₂), i in 1:length(range₁)
        initial_conditions[i, j] = [range₁[i], range₂[j], 0., 0.]
    end
    return initial_conditions
end

function solve_double_pendulum_ensemble(u₀s, p;tspan = (0.,100.), abstol=1e-15, reltol=1e-15, tstep=0.1)
    prob = ODEProblem(double_pendulum!, u₀s[1], tspan, p, saveat=0.1)
    function prob_func(prob,i,repeat)
        remake(prob,u0=u₀s[i])
    end

    ensembleproblem = EnsembleProblem(prob, prob_func=prob_func)
    sol = solve(ensembleproblem, Vern9(), EnsembleThreads(), trajectories=length(u₀s))
end

function first_flip_time(range₁, range₂, sol)
    output₁ = zeros(length(range₁), length(range₂))
    output₂ = similar(output₁)
    Threads.@threads for i in 1:length(sol)
        θ₁ = @view sol[:,i][1,:]
        θ₂ = @view sol[:,i][2,:]
        index₁ = 1
        for j in 1:length(θ₁)
            if θ₁[index₁] > π || θ₁[index₁] < -π
                output₁[i] = sol[:,i].t[index₁]
                break
            else
                index₁ += 1
            end
        end
        index₂ = 1
        for j in 1:length(θ₂)
            if θ₂[index₂] > π || θ₂[index₂] < -π
                output₂[i] = sol[:,i].t[index₂]
                break
            else
                index₂ += 1
            end
        end
    end
    return (output₁, output₂)
end

range₁ = Array(-3:0.01:3)
range₂ = Array(-3:0.01:3)

u₀s = initialise_u₀(range₁, range₂)

@info "starting ODE solver"
sol = solve_double_pendulum_ensemble(u₀s, p)

@info "calculating flip time"
firstfliptime = first_flip_time(range₁, range₂, sol)

PATH = pwd()
ds = NCDataset(joinpath(PATH, "Output", "first_flip_time.nc"), "c")

defDim(ds,"theta_1s", length(range₁))
defDim(ds,"theta_2s", length(range₂))

ds.attrib["title"] = "First-Flip Time for the Double Pendulum"

theta_1 = defVar(ds, "theta_1", range₁, ("theta_1s",))
theta_2 = defVar(ds, "theta_2", range₂, ("theta_2s",))

first_flip_time_1 = defVar(ds, "first flip time 1", firstfliptime[1], ("theta_1s", "theta_2s"))

first_flip_time_2 = defVar(ds, "first flip time 2", firstfliptime[2], ("theta_1s", "theta_2s"))

close(ds)
