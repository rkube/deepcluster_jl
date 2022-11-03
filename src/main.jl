using Clustering
using Flux
using Flux.Data: DataLoader
using MLDatasets
using Clustering
using CairoMakie

"""
Basic implementation of 
Deep Clustering for Unsupervised Learning of Visual Features
    M. Caron et al. (2019)
"""

# Load MNIST as a training set
all_img_x, all_y = MNIST(:train)[:]
all_img_x = Flux.unsqueeze(all_img_x, 3);

n_samples = length(all_y)
# Get ground-truth assignments for MNIST training set
struct GroundTruthResult <: ClusteringResult
    assignments::Vector{Int}   # assignments (n)
end
truth = GroundTruthResult(all_y .+ 1)
# Count cluster labels assignments
class_counts_truth = [(i, count(==(i), truth.assignments)) for i ∈ 1:10]
P_truth = [count(==(i), truth.assignments) for i ∈ 1:10] ./ n_samples

batch_size = 128
code_length = 64

# Initialize a CNN for classification
model = Chain(
    # 1st convolution 28x28 => 14x14
    Conv((3, 3), 1 => 8, pad=(1, 1), relu),
    x -> maxpool(x, (2, 2)),
    # 2nd convolution: 14x14 => 7x7
    Conv((3, 3), 8=>16, pad=(1, 1), relu),
    x -> maxpool(x, (2, 2)),
    # 3rd convolution: 7x7 => 3x3
    Conv((3, 3), 16 => 32, pad=(1, 1), relu),
    x -> maxpool(x, (2, 2)),
    # Reshape, use : to collate x,y,and Channels
    x -> reshape(x, 288, size(x)[4]),
    Dense(288 => 64),
    #Dropout(0.1), 
    x -> relu(x),
    Dense(64 => code_length),
    #Dropout(0.1),
    x -> relu(x), 
    Dense(code_length, 10)
)

# Store CNN code for entire dataset
#features = zeros(code_length, size(all_img_x)[end])


opt = Descent(0.01);
params = Flux.params(model);
# Run the dataset through the model.
# Option: Pass the training data batch-wise through the model.
# for (b, X) in enumerate(data_loader)
#     # Run X through layers 1-8 of model. This corresponds to
#     # features = compute_features(dataloader, model, len(dataset))
#     code = model[1:8](X)
#     @show size(code)
#     features[:, (b-1) * batch_size + 1:b * batch_size, :] .= code[:, :]
# end

# Cluster the code
NMI_list = zeros(100)
for i ∈ 1:100
    features = model[1:10](all_img_x);
    cluster_code = kmeans(features, 10, maxiter=500, display=:iter)
    # The total cost is can be computed as
    #all_loss = 0.0
    #for ix ∈ 1:60_000
    #    all_loss += norm(features[:, ix]- cluster_code.centers[:, cluster_code.assignments[ix]])^2
    #end
    #all_loss ≈ cluster_code.totalcost

    # Construct a new dataloader.
    data_loader = DataLoader((all_img_x, cluster_code.assignments); batchsize=batch_size, shuffle=true)

    # Train the model in the data.
    for (X, Y) in data_loader

        grads = Flux.gradient(params) do 
            batch_loss = Flux.Losses.crossentropy(softmax(model(X)), Flux.onehotbatch(Y, 1:10))
            @show batch_loss
        end
        Flux.Optimise.update!(opt, params, grads)
    end

    ### Now evaluate the normalized mutual information
    class_counts_K = [(i, count(==(i), cluster_code.assignments)) for i ∈ 1:10];
    # Class probability
    P_K = [count(==(i), cluster_code.assignments) for i ∈ 1:10] ./ n_samples;

    # Build the association matrix.
    N = counts(truth, cluster_code)

    # Probability that assignment of ground-truth is in class 1:10. Varies along rows. Fixed along columns
    P_i = reshape(repeat([sum(truth.assignments .== i) for i ∈ 1:10] ./ n_samples, outer=10), 10, 10);
    # # Probability that assignment of K-means is in class 1:10. Varies along column. Fixed along rows
    # #P_j = reshape(repeat([sum(res_K.assignments .== i) for i ∈ 1:10] ./ 60000, inner=10), 10, 10)
    P_j = reshape(repeat([sum(truth.assignments .== i) for i ∈ 1:10] ./ n_samples, inner=10), 10, 10);

    P_ij = N / n_samples .+ 1e-16;

    # Entropy of the ground-truth
    H_truth = -sum(P_truth .* log.(P_truth));
    # # Entropy of clustering
    H_K = -sum(P_K .* log.(P_K));
    # # Cross-entropy of ground truth and K-means
    I = sum(P_ij .* log.(P_ij ./ P_i ./ P_j));
    # # Calculate normalized Mutual Information
    # # N. Vinh et al. - Information Theoretic Measures for Clustering Comparison (2009)
    # The VI is lower bounded by 0 (when the two clusterings are identical) and always upper bounded by
    #log(N ), though tighter bounds are achievable depending on the number of clusters (
    @show NMI = I / sqrt(H_truth * H_K);
    NMI_list[i] = NMI;
end
f, a, l = lines(NMI_list)