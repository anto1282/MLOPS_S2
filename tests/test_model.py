from my_project.model import MyAwesomeModel

import torch
@pytest.mark.parametrize("batch_size",[32,64,128])
def test_model():
    model = MyAwesomeModel()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)