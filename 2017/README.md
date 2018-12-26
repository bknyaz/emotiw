# EmotiW 2017

## Examples of predictions on videos

| | | | | | |
| :-------------: |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| <figure> <img src="examples/angry.png" height="128"><figcaption>Angry</figcaption></figure> | <figure> <img src="examples/fear.png" height="128"><figcaption>Fear</figcaption></figure> | <figure> <img src="examples/happy.png" height="128"><figcaption>Happy</figcaption></figure> | <figure> <img src="examples/neutral.png" height="128"><figcaption>Neutral</figcaption></figure> | <figure> <img src="examples/sad.png" height="128"><figcaption>Sad</figcaption></figure> | <figure> <img src="examples/surprise.png" height="128"><figcaption>Surprise</figcaption></figure> |


## Description

Run the code as: ```python3 submission5.py```

The code will download all necessary features precomputed using our models, train emotion classification models and predict emotions on the test set.

Test accuracy is 60.03% (second place).

### Requirements
- Python3 (necessary to load features saved in python3)

### Test environment
- Ubuntu 14.04 LTS

## References

[Challenge website](https://sites.google.com/site/emotiwchallenge/home)

Paper:
[`Convolutional neural networks pretrained on large face recognition datasets for emotion classification from video`](https://arxiv.org/abs/1711.04598) accepted to the 1st	Workshop on Large-scale	Emotion
Recognition	and	Analysis. IEEE	International	Conference	on	Automatic	Face	and	Gesture
Recognition	2018, Xi'an, China

If you use our code or features in your work, please cite our paper as following:
```
@inproceedings{knyazev2018leveraging,
  title={Leveraging large face recognition data for emotion classification},
  author={Knyazev, Boris and Shvetsov, Roman and Efremova, Natalia and Kuharenko, Artem},
  booktitle={Automatic Face \& Gesture Recognition (FG 2018), 2018 13th IEEE International Conference on},
  pages={692--696},
  year={2018},
  organization={IEEE}
}
```
