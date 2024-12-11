# Face Swapping AI

This project performs face swapping between two images using OpenCV and dlib. The script detects faces, aligns them, warps the source face to match the target face, and blends the images seamlessly.

## Requirements

- Python 3.x
- OpenCV
- dlib
- NumPy

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/face-swapping-ai.git
    cd face-swapping-ai
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Download the dlib face landmark model:
    - [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
    - Extract the `.bz2` file and place `shape_predictor_68_face_landmarks.dat` in the project directory.

## Usage

1. Place your images in the project directory.
2. Run the script:
    ```sh
    python main.py
    ```
3. Enter the paths for the two images when prompted.

## Example

```sh
Enter the path for the first image: test1.jpg
Enter the path for the second image: [test2.jpg](http://_vscodecontentref_/0)
```

The result will be saved as result.jpg in the project directory.