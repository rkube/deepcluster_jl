using Random
using CairoMakie
using ColorSchemes
using Hungarian
using Clustering: counts

export plot_mnist_classified, plot_kstar_ecei_clf, plot_kstar_ecei_wrapped

function plot_mnist_classified(images, class_assignments, epoch; num_classes=10)
    # model - vector of images, shape (28x28xnum_samples)
    # class_assignment - vector of class assignments
    # epoch (to set in title)

    # Randomly shuffle the vectors first
    idx = randperm(length(class_assignments))
    images = images[:, :, :, idx];
    class_assignments = class_assignments[idx];

    img_all_classes = zeros(eltype(images), (28*num_classes, 28 * num_classes));

    for c ∈ 1:num_classes
        # Find where class_assignment is the current class and add the image to the correct row 
        # in img_all_classes
        counter = 1
        for idx ∈ findall(class_assignments .== c)
            counter > 10 && break
            img_all_classes[((c-1) * 28 + 1): (c * 28), ((counter - 1) * 28 + 1):(counter * 28)] = (images[:, end:-1:1, 1, idx])
            counter += 1
        end
    end
    f, a, p = heatmap(img_all_classes, colormap=:grays)
end


"""
    Plot time series with predicted labels coded by color.
    Before being used, the predicted labels are re-assigned to a class permutation
    that is calculated from the true labels using the Hungarian algorithm.

"""

function plot_kstar_ecei_clf(signal, tbase, labels_pred, labels_true; epoch=1)
    # Set up confusion matrix to find optimal permutations from labels_pred on labels_true
    cm = counts(labels_pred, labels_true)
    cm2 = -cm .+ maximum(cm)
    ix_perm = [findfirst(Hungarian.munkres(cm2)[i, :].==Hungarian.STAR) for i = 1:3]

    # Permute the indices
    pred_shuffled = [ix_perm[i] for i ∈ labels_pred.assignments]
    
    colors_pred = ColorSchemes.Accent_3[pred_shuffled];
    colors_true = ColorSchemes.Accent_3[labels_true.assignments]


    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="time [s]", title = "Epoch $(epoch)")
    lines!(ax, tbase[2:end-1], signal[12, 4, 2:end-1], color=colors_pred)
    lines!(ax, tbase[2:end-1], signal[12, 4, 2:end-1] .+ abs(minimum(signal[12, 4, 2:end-1])), color=colors_true)

    fig
end

"""
    Plot time series with predicted labels, coded by color.
    Use for models that wrap multiple frames into a feature
"""
function plot_kstar_ecei_wrapped(signal, labels_pred, labels_true, wrap_frames; epoch=1)
    # Set up confusion matrix to find optimal permutations from labels_pred on labels_true
    cm = counts(labels_pred, labels_true)
    cm2 = -cm .+ maximum(cm)
    ix_perm = [findfirst(Hungarian.munkres(cm2)[i, :].==Hungarian.STAR) for i = 1:2]

    # Permute the indices
    pred_shuffled = [ix_perm[i] for i ∈ labels_pred.assignments]
    # Unfold frame wrapping

    colors_pred = repeat(ColorSchemes.Accent_3[pred_shuffled], inner=wrap_frames)
    colors_true = repeat(ColorSchemes.Accent_3[labels_true.assignments], inner=wrap_frames)

    num_plot = length(colors_pred)

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="sample index", title = "Epoch $(epoch)")
    lines!(ax, 1:num_plot, signal[12, 4, 1:num_plot], color=colors_pred)
    lines!(ax, 1:num_plot, signal[12, 4, 1:num_plot] .+ abs(minimum(signal[12, 4, :])), color=colors_true)

    fig
end
