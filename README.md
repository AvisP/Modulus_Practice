# Modulus_Practice
Repo for replicating the modulus examples as shown [here](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/foundational/1d_wave_equation.html). ALl the examples can be run by installing `modulus` packagae `nvidia-modulus.sym` package. 
```
pip install nvidia-modulus
pip install "pint==0.19.2"
pip install nvidia-modulus.sym --no-build-isolation
```
**Note** : Works on linux, for windows use wsl and setup cuda drivers, cuda toolkit and suitable version for torch that matches cude driver before installing modulus

After successful installation each of the repos can be run by executing the `Solver.py` and it will create an `outputs` folder. After wards tensorboard can be launched and the training information viewed by running the command `tensorboard --logdir=./`
The script contains modification to visualize the plots that is not available in the original ones.
### Example 1 
This is based on the 1D [example](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/foundational/1d_wave_equation.html) of wave propagation

![Ex1DemoVideo](https://github.com/user-attachments/assets/fd183da1-802e-4ac3-b6f2-90cb1feac040)

### Example 2
This is based on the 2D wave propagation [example](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/foundational/2d_wave_equation.html)
![Ex2DemoVideo](https://github.com/user-attachments/assets/57f68d92-4c2a-4874-b111-1cdfd26fb3d6)

### Example 3
This is based on the Lid Driven Caity [example](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/foundational/zero_eq_turbulence.html) with Zero Equation Turbulence

### Example 4
This is based on the 2D heat equation [example](https://docs.nvidia.com/deeplearning/modulus/modulus-sym/user_guide/foundational/scalar_transport.html)

### Bioreactor (Work in Progress)
Implementation of a Physics Informed Neural network of a simplified Bio Reactor Model 
