using Random
using CairoMakie

export plot_mnist_classified

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
