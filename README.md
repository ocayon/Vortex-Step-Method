# Vortex Step Method
Implementation of the Vortex Step Method for a static wing shape.

## Installation Instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/ocayon/Vortex-Step-Method
    ```

2. Navigate to the repository folder:
    ```bash
    cd Vortex-Step-Method
    ```
    
3. Create a virtual environment:
   
   Linux or Mac:
    ```bash
    python3 -m venv venv
    ```
    
    Windows:
    ```bash
    python -m venv venv
    ```
    
5. Activate the virtual environment:

   Linux or Mac:
    ```bash
    source venv/bin/activate
    ```

    Windows
    ```bash
    .\venv\Scripts\activate
    ```

6. Install the required dependencies:

   For users:
    ```bash
    pip install .
    ```
        
   For developers:
    ```bash
    pip install -e .[dev]
    ```

7. To deactivate the virtual environment:
    ```bash
    deactivate
    ```

## Contributing Guide
We welcome contributions to this project! Whether you're reporting a bug, suggesting a feature, or writing code, here’s how you can contribute:

1. Create an issue on GitHub
2. Create a branch from this issue and change the branch source to `develop`
3. Use provided cmds to checkout this branch locally
4. --- Implement your new feature---
5. Verify nothing broke using pytest
```
  pytest
```
7. git add, git commit (with # to current Issue number), git push
```
  git add .
  git commit -m "#<number> <message>"
  git push
```
7. Create a pull-request, with `base:develop`, to merge this feature branch and close this issue
9. Update branch information locally using `git fetch --prune`, pull in new info `git pull origin develop` and delete branch locally using `git branch -d <enter branch name>`
```
  git fetch --prune
  git pull --all
  git checkout develop
  git pull
```
9. Once merged on the remote and locally, delete this feature branch on the remote (see pull-request) and locally using 
```
  git branch -d <branch name>
```
10. Close issue


### Code Style and Guidelines
- Follow PEP 8 for Python code style.
- Ensure that your code is well-documented.
- Write unit tests for new features and bug fixes.

## Citation
If you use this project in your research, please consider citing it. Citation details can be found in the `CITATION.cff` file included in this repository.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Copyright
Copyright (c) 2022 Oriol Cayon
Copyright (c) 2024 Oriol Cayon, Jelle Poland, TU Delft
