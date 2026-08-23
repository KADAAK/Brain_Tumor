import numpy as np
from processing.measurements import measure_component


def test_measurement_without_spacing_is_explicitly_unavailable():
    mask=np.zeros((20,20),bool); mask[5:10,7:12]=True
    result=measure_component(mask)
    assert result["area_pixels"] == 25
    assert result["width_pixels"] == 5
    assert result["physical_available"] is False


def test_measurement_uses_spacing():
    mask=np.zeros((10,10),bool); mask[1:3,1:4]=True
    result=measure_component(mask,(2.0,1.0))
    assert result["area_mm2"] == 12.0
