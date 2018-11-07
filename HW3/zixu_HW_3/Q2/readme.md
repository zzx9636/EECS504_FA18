# EECS 504 Fall 2018 HW 3 Problem 2

Find the name of cathedral in the given image through crawlering Flicker.


### Prerequisites

```
Python 2
Opencv
matplotlib
numpy
```

We need to have input image named as `hw3_img.jpg` and church list `list_church.txt` in the same directory.

## Running the tests


### Full Test: Flicker Crawler + Image Matching
By default, image links from crawler will be saved in `./list_json`.

```
python2 main.py --query list_church.txt
```
### Test Image Matching
Since crawlering takes a lot of time, we provide a list of json files from crawlering.

```
python2 main.py --query list_church.txt --json ./list_json
```

## Acknowledgments

* Flicker Crawler based on Haozhi Qi's [code from github](https://github.com/Oh233/Flicrawler.git). Modified for this question.
