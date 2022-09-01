# DJtransGAN: Automatic DJ Transitions with Differentiable Audio Effects and Generative Adversarial Networks (Data Generation Pipeline)

> This repository contains the code for "[Automatic DJ Transitions with Differentiable Audio Effects and Generative Adversarial Networks](https://arxiv.org/abs/2110.06525)"
> *2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2022)*
> Bo-Yu Chen, Wei-Han Hsu, Wei-Hsiang Liao, Marco A. Martínez-Ramírez, Yuki Mitsufuji, Yi-Hsuan Yang

## Overview

This repo contain the DJtransGANs' code of data generation pipeline which is the essential part of  DJtransGAN. After the dataset generaing, please refer to [DJtransGAN](https://github.com/ChenPaulYu/DJtransGAN) repo to train the model. The completed pipeline contain four part 

1. Feature extraction: beat & downbeat tracking, key estimation and structure boundary detection 
2. Mixability estimation: use muscial rule and neural network to find the two mixabile pair. 
3. Alignment: bpm, key and cue region aligment to make sure to segment can mix perfectly. 


Furthermore, if you want to hear more audio example, please check our demo page [here](https://paulyuchen.com/djtransgan-icassp2022/).


## Dataset

We collected two datasets to train our proposed approaches: DJ mixset from [Livetracklist](https://www.livetracklist.com/) and individual EDM tracks from [MTG-Jamendo-Dataset](https://github.com/MTG/mtg-jamendo-dataset). 

To be more specific, We in-house collect long DJ mixsets from Livetracklist and only consider the mixset with mix tag to ensure the quality of the mixset. Furthermore, we select the individual EDM track from MTG-Jamendo-Dataset, which means the track with an EDM tag in the total collection. The detailed information about using these two datasets to train our model is described in Section 3.1; please check it if you are interested. 

Last, we can not provide our training dataset for reproducing our results because of the license issues. However, we release the training code and pre-trained model for reprocude and more applicable in [DJtransGAN](https://github.com/ChenPaulYu/DJtransGAN).


## Setup 

### Install
```

pip install -r requirements.txt

```

### Set up extenal package
We use[ music puzzle game](https://github.com/remyhuang/music-puzzle-games) to do the mixability estimation, thus you need to clone it in the begining. 

```

git clone https://github.com/remyhuang/music-puzzle-games.git music_puzzle_games

```

### Configuration
Next, you should set the configuration in `pipeline/config/settings.py`  for global usage of the repo, Most important of all, you should set the path of `TRACK_DIR`, `MIX_DIR` and  `STORE_DIR`.

1. `TRACK_DIR` :  the directory conatin the collection of EDM tracks, you should put all your EDM track under the `{TRACK_DIR}/audio`. 
2. `MIX_DIR` : the directory conatin the collection of mix and its cue point, you should put all you mix under the `{MIX_DIR}/audio` and provide a meta data file `{TRACK_DIR}/meta.json` include the cue point of individual mix (we provide a sample in the repo for you to check the format). 


## Usage

We release several usage examples in `examples/` and `script/` for data generation and the usage of invidual step in pipeline. please check it, if you want to use or modify it.

### Mixable pair generation 
To generate the mixabile, you need to run two script sequentially. First, you should run the script in `script/create_segment.py` which is going to segment the collection of EDM track to several music segment. 

```
python create_segment.py [--feature=(bool, ex: 1)] [--stem=(bool, ex: 1)] [--segment=(bool, ex:1)] [--n_core=(int, ex: 5)] [--n_gpu=(int, ex: 0)]

```

- `--stem` :   specify whether to first cache the source separtion result for instrument detection. 
- `--feature` :  specify whether to first cache the feature extraction result.  
- `--n_core` : the number of mutlti-processor you want to use
- `--n_gpu` :  the number of gpu you want to use 

 

Next, you need to run the script in  `script/create_pair.py` to match the music segment which is sutible to mix together. 

```
python create_pair.py [--match=(str, e.g: None, all, nn, rule)] [--n_sample=(int, ex: 10)] [--n_core=(int, ex: 4)]

```

- `--match` :  speicify the mixabiltiy estimaiton approach  
	- None: random pick one segment as best candidate
	- rule: use muscial rule to filter out the data and random pick one segment as best candidate
	- nn: use neural netwrok (SEN form [music puzzle game](https://github.com/remyhuang/music-puzzle-games)) to pick one segemnt as best candidate
	- all: use muscial rule to filter out the data and use neural network to pick one segment as best candaite from filted data. 
- `--n_core` :  the number of mutlti-processor you want to use
- `--n_sample` :  the maxium number of sample for mixability estimation (speed up the training )



### Mix generation
To train the DJtransGAN, we still need a professional DJ mix as reference for GAN to learn. Thus, you can run the script in `script/create_mix.py` to get such dataset. 


```
python create_mix.py [--n_core=(int, ex: 4)]

```
- `--n_core` : the core number of multi-processor to speed up the process



## Citation

If you use any of our code in your work please consider citing us.

```
  @inproceedings{chen2022djtransgan,
    title={Automatic DJ Transitions with Differentiable Audio Effects and Generative Adversarial Networks},
    author={Chen, B. Y., Hsu, W. H., Liao, W. H., Ramírez, M. A. M., Mitsufuji, Y., & Yang, Y. H.},
    booktitle={ICASSP},
    year={2022}}
```

## Acknowledgement

This repo is done during the internship in the Sony Group Corporation with outstanding mentoring by my incredible mentors in Sony [Wei-Hsiang Liao](https://jp.linkedin.com/in/wei-hsiang-liao-66283154), [Marco A. Martínez-Ramírez](https://m-marco.com/), and [Yuki Mitsufuji](https://www.yukimitsufuji.com/) and my colleague [Wei-Han Hsu](https://github.com/ddman1101) and advisor [Yi-Hsuan Yang](https://www.citi.sinica.edu.tw/pages/yang/) in Academia  Sinica. The results are a joint effort with Sony Group Corporation and  Academia  Sinica. I sincerely appreciate all the support made by them to make this research happen. Moreover, please check the other excellent AI research made by Sony [here](https://github.com/sony/ai-research-code) and their recent work ["FxNorm-automix"](https://marco-martinez-sony.github.io/FxNorm-automix/) and ["distortionremoval"](https://joimort.github.io/distortionremoval/) which is going to present in ISMIR 2022. 



## License
Copyright © 2022 Bo-Yu Chen

Licensed under the MIT License (the "License"). You may not use this
package except in compliance with the License. You may obtain a copy of the
License at

    https://opensource.org/licenses/MIT

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
