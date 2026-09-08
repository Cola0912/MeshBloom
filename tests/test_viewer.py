import numpy as np
import pytest
from treesupport.viewer import Camera


def test_fit_pan_zoom_and_pole_views():
    camera = Camera()
    bounds = np.array([[10., 20., 0.], [30., 40., 50.]])
    camera.fit(bounds, .5)
    np.testing.assert_allclose(camera.target, [20, 30, 25])
    size = camera.half_height
    camera.zoom(2)
    assert camera.half_height < size
    camera.zoom(-2)
    assert camera.half_height == pytest.approx(size)
    before = camera.target.copy()
    camera.pan(10, 0, 400)
    assert np.linalg.norm(camera.target-before) == pytest.approx(10*2*size/400)
    for elevation in [-90, 0, 90]:
        camera.elevation = elevation
        axes = np.array(camera.basis())
        np.testing.assert_allclose(axes@axes.T, np.eye(3), atol=1e-12)
    camera.orbit(10000, 10000)
    assert 0 <= camera.azimuth < 360 and abs(camera.elevation) < 90
    for _ in range(100):
        camera.zoom(20)
    assert camera.half_height > 0
    camera.fit(bounds)
    np.testing.assert_allclose(camera.target, before)
