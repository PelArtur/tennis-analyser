# Tennis Analyser

The **tennis analyser** is a system of deep learning networks designed to extract statistics from input video footage. This system identifies the location of the ball, players, and court markings in the video and processes this data to generate statistics such as player movement heatmaps, ball trajectory, ball speed, and more.

This project was developed as part of the Artificial Intelligence course during the third year of the Computer Science program at the Ukrainian Catholic University.

<table>
    <tr>
        <td>
            <video width="320" height="240" controls>
                <source src="./videos/video1.mp4" type="video/mp4">
            </video>
        </td>
        <td>
            <video width="320" height="240" controls>
                <source src="./videos/video2.mp4" type="video/mp4">
            </video>
        </td>
    </tr>
</table>



## Our Team
- [Artur Pelcharskyi](https://github.com/PelArtur)
- [Iryna Kokhan](https://github.com/ironiss)
- [Anna Yaremko](https://github.com/moisamidi)


## Prerequirements
- Our repository contains weight files for most of the models used; however, **not all files could be uploaded** due to GitHub restrictions. Therefore, to ensure functionality, you will need to download all the required files from this [link](https://drive.google.com/file/d/1S5Dh8J6LXWf2SBCrmQjWt3IpOaVvwtiC/view?usp=sharing).
- To ensure that all libraries function correctly, you need to use **Python** version `3.12.X`.
- For optimal performance, we recommend running our project in a **GPU**-enabled environment.

## How to run
1. Clone this repository
  
    ```git
    git clone https://github.com/PelArtur/tennis-analyser
    ```
    
2. Install the requirements using pip 

    ```python
    pip install -r requirements.txt
    ```
  
3. Run the following command in the command line
  
    ```shell
    python3 process_video.py -i ./videos/video1.mp4 -o ./output_video.moy
    ```

