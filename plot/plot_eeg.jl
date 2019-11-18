
using Plots
using ProgressMeter

function eeg_animate(eeg, n, idx, skip=1; kwargs...)
    pr = ProgressMeter.Progress(Int64(floor((length(eeg.activations)-n)/skip)), 0.01, "Anim: ")
    anim = @animate for i in 1:skip:(length(eeg.activations)-n)
        l = @layout grid(length(eeg.activations[1]), 1)
        plt_a = plot(
            hcat(getindex.(eeg.activations, 1)...)',
            xlims=(i, n+i), title="Inputs", label="")
        plt_acts = [plot(
            hcat(getindex.(eeg.activations, idx)...)',
            xlims=(i,n+i), label="", title="Layer $(idx)") for idx in 2:length(eeg.activations[1])]
        plot(plt_a, plt_acts...; layout=l, kwargs...)
        next!(pr)
        # gui()
    end
    anim
end

function plot_eeg(eeg)


end

