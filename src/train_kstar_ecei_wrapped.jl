#
using Augmentor
using CairoMakie
using Clustering: mutualinfo, ClusteringResult
using CUDA
using Flux
using Flux.Data: DataLoader
using LinearAlgebra
using MLUtils
using MultivariateStats
using ParallelKMeans
using Statistics
using Zygote


using deepcluster_jl

push!(LOAD_PATH, "/home/rkube/repos/kstar_ecei_data")
using kstar_ecei_data

"""
Train Deep Clustering on single KSTAR ECEI shot
Shot should have ELM filaments and ELM crash visible. So there are two classes.
"""


struct GroundTruthResult <: ClusteringResult
    assignments::Vector{Int}   # assignments (n)
end

# For weight regularization
sqnorm(x) = sum(abs2, x);

shotnr = 26512
wrap_frames = 8

num_classes = 2
num_epochs = 20
batch_size = 32
code_length = 64

lr = 1e-3
λ = 1f-4


trf = GaussianBlur(3)
my_ds = kstar_ecei_3d(shotnr, wrap_frames, trf)

# Loads training data
loader = DataLoader(my_ds, batchsize=batch_size, shuffle=true, partial=false)
# Load all data, calculate histogram and move to gpu.
loader_all = DataLoader(my_ds, batchsize=size(my_ds.features, 4), shuffle=false, partial=false)
# Move vector of all data to gpu. Prediction of this are evaluated during training
(x_all, labels_true) = first(loader_all)
# fix for num_classes = 2: ELMcrash is the same as filament
labels_true[labels_true .== 2] .= 1;


assignments_true = GroundTruthResult(labels_true .+ 1)
f, a, h = hist(x_all[:], bins=-2.5:0.01:2.5);
save("plots/hist_x_all.png", f)


x_all = Flux.unsqueeze(x_all, 4);
# Used in plotting later
x_all_flat = reshape(x_all, 24, 8, size(x_all, 3) * size(x_all, 5))  |> cpu;

model = Chain(Conv((5, 3, 3), 1 => 16),   # 
              BatchNorm(16, relu),
              Conv((7, 3, 3), 16 => 64),  # 
              BatchNorm(64, relu),
              # Put in some skip connections
              SkipConnection(Chain(Conv((3, 3, 3), 64 => 64, pad=(1,1,1)),
                                   BatchNorm(64, relu),
                                   Conv((3, 3, 3), 64 => 64, pad=(1,1,1)),
                                   BatchNorm(64, relu)),
                             +),
              # reduce size from (14, 4, 4) to (8, 2,2)
              Conv((7, 3, 3), 64 => 64),
              SkipConnection(Chain(Conv((3, 3, 3), 64 => 64, pad=(1,1,1)),
                                   BatchNorm(64, relu),
                                   Conv((3, 3, 3), 64 => 64, pad=(1,1,1)),
                                   BatchNorm(64, relu)),
                             +),

              BatchNorm(64, relu),
              Conv((5, 1, 1), 64 => 64),
              x -> Flux.flatten(x),
              Dense(4 * 2 * 2 * 64, code_length),
              x -> relu(x),  # Keep ReLU separate. The classifier for K-Means is called without ReLU...
                             # So the ReLU call has to be separable by slicing operation 
              Dense(code_length, num_classes)
) |> gpu;



ix_model_backbone = 11; #model[1:11] calls just the backbone of the model
#opt = Flux.Optimise.Optimiser(Momentum(lr), ExpDecay(1.0, 0.1, 5, 1e-4));
opt = ADAM(lr);

# Cluster the code
NMI_list = zeros(num_epochs);
# We are oversampling and don't know the number of batches a-priori after over-sampling. Take a large number and cross fingers
batch_losses = zeros(20000, num_epochs);

for epoch ∈ 1:num_epochs
    # Use all model layers but exclude the last relu and the final dense layer
    features = model[1:ix_model_backbone](x_all) |> cpu;
    # Use whitened features to improve kmeans performance
    trf_w = fit(Whitening, features);
    features_white = transform(trf_w, features);

    cluster_code = kmeans(features_white, num_classes); #, maxiter=500; display=:iter);
    class_counts = [(i, count(==(i), cluster_code.assignments)) for i ∈ 1:num_classes]
    assignments_pred = GroundTruthResult(cluster_code.assignments)

    # Use completely random labels in the first epoch
    epoch == 1 ? assignments_pred = GroundTruthResult(rand(1:num_classes, length(assignments_pred))) : nothing
    
    NMI_list[epoch] = mutualinfo(assignments_pred, assignments_true)
    @show epoch, NMI_list[epoch], class_counts

    # Plot current clustering result.
    fig = plot_kstar_ecei_wrapped(x_all_flat, assignments_pred, assignments_true, wrap_frames; epoch=epoch)
    save("plots/kstar_assignments_$(epoch).png", fig)

    # Over-sample the images so that each cluster has an identical amount of samples for training
    data_os, labels_os = oversample((my_ds.features, assignments_pred.assignments));
    # Create full arrays with the oversampled data. This runs faster than repeated calls to oversampled array.
    data_os_gpu = copy(data_os);
    data_loader = DataLoader((data_os_gpu, labels_os); batchsize=batch_size, shuffle=true)

    ### We need to re-initialize the weights of the top layer here.
    model[end].weight[:, :] .= Flux.glorot_uniform(size(model[end].weight)...) |> gpu;
    model[end].bias .= 0f0
    params = Flux.params(model);

    # Train the model in the data.
    ix = 1
    for (X, Y) in data_loader
        X = Flux.unsqueeze(X, 4) |> gpu;
        Y_1h = Flux.onehotbatch(Y, 1:num_classes) |> gpu;


        loss, back = Zygote.pullback(params) do 
            Flux.Losses.crossentropy(softmax(model(X)), Y_1h) + λ * sum(sqnorm, params);
        end
        grads = back(one(loss))
        Flux.Optimise.update!(opt, params, grads)
        batch_losses[ix, epoch] = loss;
        ix += 1
    end

    #@show opt[1].eta, epoch
    plot_idx = batch_losses[:, epoch] .> 0.0;
    f, a, l = lines(batch_losses[plot_idx, epoch]; axis=(; xlabel="batch", ylabel="loss", title="epoch $(epoch)"))
    save("plots/loss_$(epoch).png", f)
end



f, a, l = lines(NMI_list; axis=(; xlabel="epoch", ylabel="NMI"))
save("nmi.png", f)


