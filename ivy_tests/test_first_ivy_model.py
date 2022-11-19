# global
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import time
import pytest
import ivy

@pytest.mark.parametrize("backend", ["torch", "jax", "tensorflow"])
def test_first_ivy_model(backend):

    class MyModel(ivy.Module):
        def __init__(self):
            self.linear0 = ivy.Linear(3, 64)
            self.linear1 = ivy.Linear(64, 1)
            ivy.Module.__init__(self)

        def _forward(self, x):
            x = ivy.relu(self.linear0(x))
            return ivy.sigmoid(self.linear1(x))

    start = time.time()
    ivy.set_default_device("cpu")
    ivy.set_backend(backend)  # change to any framework!
    model = MyModel()
    optimizer = ivy.Adam(1e-4)
    x_in = ivy.array([1., 2., 3.])
    target = ivy.array([0.])

    def loss_fn(v):
        # TODO: fix this loss function
        # It used to be ivy.reduce_mean((out - target)**2)[0]
        out = model(x_in, v=v)
        return ivy.mean((out - target)**2)

    for step in range(100):
        loss, grads = ivy.execute_with_gradients(loss_fn, model.v)
        model.v = optimizer.step(model.v, grads)
        print('step {} loss {}'.format(step, ivy.to_numpy(loss).item()))

    ivy.unset_backend()
    elapsed_msec = (time.time() - start)*1_000
    print(f'Finished training with {backend} in {elapsed_msec:,.2f} msec.')
