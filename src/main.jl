using Clustering
using Flux
using Flux.Data: DataLoader
using MLDatasets
using MLDataPattern
using Clustering
using CairoMakie
using CUDA

"""
Basic implementation of 
Deep Clustering for Unsupervised Learning of Visual Features
    M. Caron et al. (2019)
"""

# Load MNIST as a training set
all_img_x, all_y = MNIST(:train)[:]
all_img_x = Flux.unsqueeze(all_img_x, 3);
all_img_gpu = all_img_x |> gpu;

n_samples = length(all_y);
# Get ground-truth assignments for MNIST training set
struct GroundTruthResult <: ClusteringResult
    assignments::Vector{Int}   # assignments (n)
end
truth = GroundTruthResult(all_y .+ 1);
# Count cluster labels assignments
# class_counts_truth = [(i, count(==(i), truth.assignments)) for i ∈ 1:10]
# P_truth = [count(==(i), truth.assignments) for i ∈ 1:10] ./ n_samples

# Probability that assignment of ground-truth is in class 1:10. Varies along rows. Fixed along columns
# P_i = reshape(repeat([sum(truth.assignments .== i) for i ∈ 1:10] ./ n_samples, outer=10), 10, 10);
# # # Probability that assignment of K-means is in class 1:10. Varies along column. Fixed along rows
# P_j = reshape(repeat([sum(truth.assignments .== i) for i ∈ 1:10] ./ n_samples, inner=10), 10, 10);

# Entropy of the ground-truth
# H_truth = -sum(P_truth .* log.(P_truth));

# For weight regularizatoin
sqnorm(x) = sum(abs2, x);

batch_size = 64
code_length = 32
num_batches = ceil(size(all_img_x)[end] / batch_size) |> Int

# Initialize a CNN for classification
model = Chain(
    # 1st convolution 28x28 => 26x26
    Conv((3, 3), 1 => 16, relu),
    # 2nd convolution: 26x26 => 22 x 22
    Conv((5, 5), 16 => 16, relu),
    # 3rd convolution: 22x22 => 16x16
    Conv((7, 7), 16 => 16, relu),
    # maxpool: 16x16 => 8x8
    x -> maxpool(x, (2, 2)),
    # Reshape, use : to collate x,y,and Channels
    x -> reshape(x, :, size(x)[4]),
    # Up to here the model corresponds to "features" in Caron's code
    Dropout(0.5),
    Dense(1024 => 512),
    #Dropout(0.1), 
    x -> relu(x),
    Dropout(0.5),
    Dense(512 => code_length),
    # Everything up to this line is used for clustering.
    # This corresponds to the "classifier" in Caron's code
    x -> relu(x), 
    Dense(code_length, 10)
) |> gpu;

# Store CNN code for entire dataset
#features = zeros(code_length, size(all_img_x)[end])

#opt = Flux.Optimise.Optimiser(WeightDecay(1f-4), Momentum(0.001), ExpDecay(1f0));
lr = 1e-3
λ = 1f-4
#opt = Flux.Optimise.Optimiser(Momentum(lr), ExpDecay(1.0));
opt = Momentum(lr)
params = Flux.params(model);

# Cluster the code
num_epochs = 20
NMI_list = zeros(num_epochs);
# We are oversampling and don't know the number of batchs a-priori. Take a large number and cross thumbs.
batch_losses = zeros(10000, num_epochs);

for epoch ∈ 1:num_epochs
    # Use all model layers but exclude the last relu and the final dense layer
    features = model[1:10](all_img_gpu);
    features_c = features |> cpu;
    cluster_code = kmeans(features_c, 10, maxiter=500);
    class_counts= [(i, count(==(i), cluster_code.assignments)) for i ∈ 1:10]
    NMI_list[epoch] = mutualinfo(cluster_code, truth)
    @show epoch, NMI_list[epoch], class_counts
    # Construct a new dataloader.
    #labels = Flux.onehotbatch(cluster_code.assignments, 1:10) |> gpu;
    #labels_gpu = cluster_code.assignments |> gpu;

    # Over-sample the images so that each cluster has an identical amount of
    # samples for training
    # Oversample has issues with onehotbatch.
    # In particular if we pass a onehotbatch as an argument it will return a bool matrix.
    # So instead of passing a onehotbatch, we just submit the cluster_code.assignment matrix.
    # Then convert the matrix into a one-hot batch in the iteration over the data_loader.
    img_os, labels_os = oversample((all_img_x, cluster_code.assignments));

    # Create full arrays with the oversampled data. This runs much faster
    # than repeated calls to oversampled array.
    img_os_gpu = similar(img_os);
    img_os_gpu[:] = img_os[:]
    img_os_gpu = img_os_gpu |> gpu;
    labels_full = similar(labels_os);
    labels_full[:] = labels_os[:]
    #labels_os_gpu = labels_os_gpu |> gpu;

    data_loader = DataLoader((img_os_gpu, labels_os); batchsize=batch_size, shuffle=true)

    ### We need to re-initialize the weights of the top layer here.
    model[end].weight[:,:] .= Flux.glorot_uniform(size(model[end].weight)...) |> gpu;
    model[end].bias .= 0f0

    # Train the model in the data.
    ix = 1
    for (X, Y) in data_loader
        Y_1h = Flux.onehotbatch(Y, 1:10) |> gpu;
        # Store current loss in array. This can't be in gradient call
        batch_losses[ix, epoch] = Flux.Losses.crossentropy(softmax(model(X)), Y_1h)
        #@show ix, batch_losses[ix, epoch]
        ix += 1

        grads = Flux.gradient(params) do 
            batch_loss = Flux.Losses.crossentropy(softmax(model(X)), Y_1h) + λ * sum(sqnorm, params);
        end
        Flux.Optimise.update!(opt, params, grads)
    end
    plot_idx = batch_losses[:, epoch] .> 0.0
    f, a, l = lines(batch_losses[plot_idx, epoch]; axis=(; xlabel="batch", ylabel="loss", title="epoch $(epoch)"))
    save("loss_$(epoch).png", f)

    # ### Now evaluate the normalized mutual information
    # class_counts_K = [(i, count(==(i), cluster_code.assignments)) for i ∈ 1:10];
    # # Class probability
    # P_K = [count(==(i), cluster_code.assignments) for i ∈ 1:10] ./ n_samples;
    # # Entropy of clustering
    # H_K = -sum(P_K .* log.(P_K));
    # # Build the association matrix.
    # N = counts(truth, cluster_code)
    # P_ij = N / n_samples .+ 1e-16;
    # # Cross-entropy of ground truth and K-means
    # I = sum(P_ij .* log.(P_ij ./ P_i ./ P_j));
    # Calculate normalized Mutual Information
    # N. Vinh et al. - Information Theoretic Measures for Clustering Comparison (2009)
    # The VI is lower bounded by 0 (when the two clusterings are identical) and always upper bounded by
    # log(N ), though tighter bounds are achievable depending on the number of clusters.
    # @show NMI = I / sqrt(H_truth * H_K);
    # NMI_list[epoch] = NMI;
end

f, a, l = lines(NMI_list; axis=(; xlabel="epoch", ylabel="NMI"))
save("nmi.png", f)
