# Vortex_step_method
Implementation of the Vortex Step Method for a static wing shape

# Installation
Clone the repo
```bash
git clone https://github.com/ocayon/Vortex-Step-Method
```

Go in the repo folder
```bash
cd Vortex-Step-Method

Create a new venv with the following command:


# main.py

Script to solve for one angle of attack
INPUTS:
- Coordinates of the wing (look at the rectangular wing example). 
The coordinates are defined so that the first point is the LE position of the first section, the second point the TE of the first section and so on. 
- Convergence criteria
- Velocity magnitude and direction
- Number of filaments per horseshoe (default should be 5)
- Model : 'VSM' or 'LLT'
- Airfoil polars

# References

[1] Cayon, O. Fast aeroelastic model of a leading-edge inflatable kite for the design phase of airborne wind energy systems. Master’s
thesis, Delft University of Technology, 2022. http://resolver.tudelft.nl/uuid:aede2a25-4776-473a-8a75-fb6b17b1a690

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Copyright

Copyright (c) 2022 Oriol Cayon
