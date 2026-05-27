import torch

from gaussian_model import GaussianModel
from gaussian_renderer import GaussianRenderer


def main():
    torch.manual_seed(0)
    points = torch.tensor([
        [-0.5, -0.5, 4.0],
        [0.5, -0.5, 4.2],
        [0.0, 0.5, 3.8],
        [0.0, 0.0, 5.0],
    ], dtype=torch.float32)
    colors = torch.tensor([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 255],
    ], dtype=torch.float32)

    model = GaussianModel(points, colors)
    params = model()
    cov = params["covariance"]
    assert cov.shape == (4, 3, 3)
    assert torch.isfinite(cov).all()
    assert torch.allclose(cov, cov.transpose(1, 2), atol=1e-5)

    renderer = GaussianRenderer(32, 32)
    K = torch.tensor([[30.0, 0.0, 16.0], [0.0, 30.0, 16.0], [0.0, 0.0, 1.0]])
    R = torch.eye(3)
    t = torch.zeros(3)
    means2d, covs2d, depths = renderer.compute_projection(points, cov, K, R, t)
    assert means2d.shape == (4, 2)
    assert covs2d.shape == (4, 2, 2)
    assert depths.shape == (4,)
    assert torch.isfinite(means2d).all()
    assert torch.isfinite(covs2d).all()

    image = renderer(
        params["positions"],
        params["covariance"],
        params["colors"],
        params["opacities"],
        K,
        R,
        t,
    )
    assert image.shape == (32, 32, 3)
    assert torch.isfinite(image).all()
    assert image.min() >= 0
    assert image.max() <= 1.0 + 1e-5
    print("smoke_test: PASS")


if __name__ == "__main__":
    main()
