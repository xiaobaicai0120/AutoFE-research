# AutoFE-Pointer

AutoFE-Pointer is a deep learning framework used for predicting three types of sites: 4mC, 5hmC, and 6mA.

## Content Introduction:
- **DataSets**: Stores datasets such as 4mC, 5hmC, 6mA, etc.
- **Src**: Stores the required Python script files.
- **Model-pkl**: Stores the URL links of the pkl files containing the model weights.

## Usage Steps:
**Hardware Environment**:
- NVIDIA A800 80GB  
- CUDA Version: 12.2

1. ### Install the Environment
    1.1 Create a new Python environment
    ```python
    conda create -n AutoFE python=3.9.21
    ```
    1.2 Activate the Python environment
    ```python
    conda activate AutoFE
    ```
    1.3 Enter the environment folder
    ```python
    cd AutoFE
    ```
    1.4 Install Python dependent libraries
    ```python
    pip install -r requirements.txt
    ```
2. ### Download Model Parameters
    2.1 Create a folder
    ```python
    mkdir model
    cd model
    ```
    2.2 Download the weight files of the model through the URLs in the Model-pkl folder, and place all the downloaded weight files in the model folder.
3. ### Run the Visualization Program
    3.1 Enter the specified folder
    ```
    cd Src
    ```
    3.2 Run the script of the visualization program
    ```python
    python main.py
    ``` 
