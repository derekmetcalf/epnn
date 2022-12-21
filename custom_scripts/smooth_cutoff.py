#!/usr/bin/env python
# Copyright 2022 Jesús Carrete Montaña
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cgi import test
import jax
import jax.numpy as jnp


def _aux_function_f(t):
    "First auxiliary function used in the definition of the smooth bump."
    return jnp.where(t > 0., jnp.exp(-1. / jnp.where(t > 0., t, 1.)), 0.)


def _aux_function_g(t):
    "Second auxiliary function used in the definition of the smooth bump."
    f_of_t = _aux_function_f(t)
    return f_of_t / (f_of_t + _aux_function_f(1. - t))


def smooth_cutoff(r, r_switch, r_cut):
    """One-dimensional smooth cutoff function based on a smooth bump.

    This function follows the prescription given by Loring W. Tu in
    "An Introduction to Manifolds", 2nd Edition, Springer

    Args:
        r: The radii at which the function must be evaluated.
        r_switch: The radius at which the function starts differing from 1.
        r_cut: The radius at which the function becomes exactly 0.
    """
    r_switch2 = r_switch * r_switch
    r_cut2 = r_cut * r_cut

    return 1. - _aux_function_g((r * r - r_switch2) / (r_cut2 - r_switch2))


if __name__ == "__main__":
    import functools
    import matplotlib
    import matplotlib.pyplot as plt

    R_SWITCH = 1.0
    R_CUT = 1.2
    N_DERIVATIVES = 3

    test_cutoff = functools.partial(smooth_cutoff,
                                    r_switch=R_SWITCH,
                                    r_cut=R_CUT)

    distance = jnp.linspace(0., 1.5, num=1001)
    to_plot = test_cutoff
    for order in range(N_DERIVATIVES + 1):
        samples = jax.jit(jax.vmap(to_plot))(distance)
        samples /= jnp.fabs(samples).max()
        plt.plot(distance, samples, label=f"Normalized order {order} derivative")
        to_plot = jax.grad(to_plot)

    plt.xlabel("Distance")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
