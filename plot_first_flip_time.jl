using NCDatasets
using Plots

PATH = pwd()

ds = NCDataset(joinpath(PATH, "Output", "first_flip_time_0.005.nc"))

T₁ = log10.(Array(ds["first flip time 1"]))
T₂ = log10.(Array(ds["first flip time 2"]))
θ₁ = Array(ds["theta_1"])
θ₂ = Array(ds["theta_2"])

heatmap(θ₁, θ₂, T₁)
title!("First Flip Time of Pendulum 1")
xlabel!("θ₁₀")
ylabel!("θ₂₀")

heatmap(θ₁, θ₂, T₂)
title!("lg(First Flip Time) of Pendulum 2")
xlabel!("θ₁₀")
ylabel!("θ₂₀")
savefig("Output/first_flip_time_2.pdf")