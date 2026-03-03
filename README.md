# ML-CTF

This project requires a Python virtual environment to ensure that all dependencies are properly managed. Please follow the steps below to set up the environment and run the `evaluate.py` script.

## Prerequisites
- Python 3.9.25 installed on your system.
- pip (Python package installer) should be available.
## Setup Instructions
1. **Create a Virtual Environment**:
    Open your terminal or command prompt and navigate to the directory where you want to create the virtual
    environment. Run the following command:
    ```bash
    python3 -m venv ml-ctf-env
    ```
2. **Activate the Virtual Environment**:
    - On Windows:
        ```bash
        ml-ctf-env\Scripts\activate
        ```
    - On macOS and Linux:
        ```bash
        source ml-ctf-env/bin/activate
        ```
3. **Install Required Packages**:
    Make sure you have a `requirements.txt` file in the same directory as `evaluate.py`. Run the following command to install the necessary packages:
    ```bash
    pip install -r requirements.txt
    ```

## Running the `evaluate.py` Script
Once the virtual environment is activated and the required packages are installed, you can run the `evaluate.py` script from the command line. Use the following command format:
```bash
python3 evaluate.py \
    --model_path ./models/lenet5_1.pth \ # complete path to the model
    --arch lenet5 # Option: lenet5, mini_vgg
```
