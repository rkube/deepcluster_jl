using Statistics
using LinearAlgebra
using Flux
using Flux.Data: DataLoader
using MLUtils
using Clustering: mutualinfo, ClusteringResult
using ParallelKMeans
using CairoMakie
using CUDA

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


num_classes = 3
num_epochs = 20
batch_size = 16
code_length = 32


shotnr = 26327
data_norm, tbase_norm = get_shot_data(shotnr)

num_samples = size(data_norm)[end]
# Calculate first and second derivative of image time-series manually
data_deriv1 = data_norm[:, :, 3:end] .- data_norm[:, :, 1:end-2];

data_deriv2 = data_norm[:, :, 1:end-2] .- 2f0 * data_norm[:, :, 2:end-1] .+ data_norm[:, :, 3:end];

# Stack data, first, and second derivative
data_trf = zeros(Float32, 24, 8, 3, 1, num_samples-2)
data_trf[:, :, 1, 1, :] = (data_norm[:, :, 2:end-1] .- mean(data_norm)) ./ std(data_norm);
data_trf[:, :, 2, 1, :] = (data_deriv1 .- mean(data_deriv1)) ./ std(data_deriv1);
data_trf[:, :, 3, 1, :] = (data_deriv2 .- mean(data_deriv2)) ./ std(data_deriv2);

labels_truth = get_labels(26327)
# For stacking derivatives
labels_true = labels_truth[2:end-1] .+ 1;
assignments_true = GroundTruthResult(labels_true);

data_trf = data_trf |> gpu;



model = Chain(Conv((5, 3, 3), 1 => 32, relu),   #   
              BatchNorm(32),
              MaxPool((2, 1, 1)),
              Conv((5, 3, 1), 32 => 64, relu),  #   
              BatchNorm(64),
              Conv((5, 3, 1), 64 => 64, relu),  #   
              x -> reshape(x, :, size(x)[5]),
              Dense(2 * 2 * 1 * 64 => 256, relu),
              Dropout(0.2),
              Dense(256 => code_length),
              x -> relu(x),  # Keep ReLU separate. The classifier for K-Means is called without ReLU...
                             # So the ReLU call has to be separable by slicing operation 
              Dense(code_length, num_classes)
) |> gpu;



lr = 1e-3
λ = 1f-4
opt = Flux.Optimise.Optimiser(Momentum(lr), ExpDecay(1.0));
#opt = Momentum(lr)

# Cluster the code
NMI_list = zeros(num_epochs);
# We are oversampling and don't know the number of batches a-priori after over-sampling. Take a large number and cross fingers
batch_losses = zeros(20000, num_epochs);

for epoch ∈ 1:num_epochs
    # Use all model layers but exclude the last relu and the final dense layer
    features = model[1:10](data_trf);

    # Optional: Implement PCA with whitening on the data.
    # http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/
    avg = mean(features, dims=1);
    σ = features * features' / size(features)[end];
    F = svd(σ);
    white_mat = Diagonal(1f0 ./ (sqrt.(F.S) .+ eps(eltype(F.S)))) |> gpu;
    features_white = white_mat * F.U' * features 
    features_white = features_white ./ sum(abs2, features_white, dims=1) |> cpu;

    # Better: Use whitened features to achieve better kmeans performance
    cluster_code = kmeans(features_white, num_classes); #, maxiter=500; display=:iter);
    class_counts = [(i, count(==(i), cluster_code.assignments)) for i ∈ 1:num_classes]
    assignments_pred = GroundTruthResult(cluster_code.assignments)
    NMI_list[epoch] = mutualinfo(assignments_pred, assignments_true)
    @show epoch, NMI_list[epoch], class_counts


    # Plot current clustering result.
    fig = plot_kstar_ecei_clf(data_norm, tbase_norm, assignments_pred, assignments_true; epoch=epoch)
    save("kstar_assignments_$(epoch).png", fig)

    # Over-sample the images so that each cluster has an identical amount of samples for training
    data_os, labels_os = oversample((data_trf, assignments_pred.assignments));
    # Create full arrays with the oversampled data. This runs much faster
    # than repeated calls to oversampled array.
    data_os_gpu = copy(data_os);
    data_loader = DataLoader((data_os_gpu, labels_os); batchsize=batch_size, shuffle=true)

    ### We need to re-initialize the weights of the top layer here.
    model[end].weight[:, :] .= Flux.glorot_uniform(size(model[end].weight)...) |> gpu;
    model[end].bias .= 0f0
    params = Flux.params(model);

    # Train the model in the data.
    ix = 1
    for (X, Y) in data_loader
        Y_1h = Flux.onehotbatch(Y, 1:num_classes) |> gpu;
        # Store current loss in array. This can't be in gradient call
        batch_losses[ix, epoch] = Flux.Losses.crossentropy(softmax(model(X)), Y_1h) + λ * sum(sqnorm, params);
        #@show ix, batch_losses[ix, epoch]
        ix += 1

        grads = Flux.gradient(params) do 
            batch_loss = Flux.Losses.crossentropy(softmax(model(X)), Y_1h) + λ * sum(sqnorm, params);
        end
        Flux.Optimise.update!(opt, params, grads)
    end

    @show opt[1].eta, epoch
    plot_idx = batch_losses[:, epoch] .> 0.0;
    f, a, l = lines(batch_losses[plot_idx, epoch]; axis=(; xlabel="batch", ylabel="loss", title="epoch $(epoch)"))
    save("loss_$(epoch).png", f)
end



f, a, l = lines(NMI_list; axis=(; xlabel="epoch", ylabel="NMI"))
save("nmi.png", f)


