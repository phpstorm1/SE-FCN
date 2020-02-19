# Fully Convolutional Network for Complex Spectrogram Processing in Single-channel Speech Enhancement

An implementation of paper ''*A Fully Convolutional Neural Network for Complex Spectrogram Processing in Speech Enhancement*''. The code provides a variation on training with [TED-LIUM](https://www.openslr.org/19/) dataset. 

## Dependency

- Python 3
- Tensorflow 1.15
- Scipy
- python_speech_features
- Librosa

## Usage

- Training: for training with TED-LIUM dataset, run *train_tedlium.py*, otherwise run *train.py*
- Testing: The testing part, including generating noisy speech and obtaining estimated clean speech from the model, is already a part of the training code. For testing with existing noisy wav files: run *proc_existing_noisy.py*

## Related work

- WaveNet 
- TED-LIUM (speech corpus)
- ESC-50 (environmental sound dataset / noise dataset)

## Reference

- This model

  ```BibTex
  @inproceedings{ouyang2019fully,
    title={A fully convolutional neural network for complex spectrogram processing in speech enhancement},
    author={Ouyang, Zhiheng and Yu, Hongjiang and Zhu, Wei-Ping and Champagne, Benoit},
    booktitle={ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    pages={5756--5760},
    year={2019},
    organization={IEEE}
  }
  ```

- TED-LIUM (v2)

  ```BibTex
  @inproceedings{rousseau2014enhancing,
    title={Enhancing the TED-LIUM corpus with selected data for language modeling and more TED talks.},
    author={Rousseau, Anthony and Del{\'e}glise, Paul and Esteve, Yannick},
    booktitle={LREC},
    pages={3935--3939},
    year={2014}
  }
  ```

- ESC-50

  ```BibTeX
  @inproceedings{piczak2015esc,
    title={ESC: Dataset for environmental sound classification},
    author={Piczak, Karol J},
    booktitle={Proceedings of the 23rd ACM international conference on Multimedia},
    pages={1015--1018},
    year={2015}
  }
  ```

## Notes

In this implementation, the model is slightly modified from the one in paper.

## TODO

- [ ] Migrate to Tensorflow 2
- [ ] A detailed README
- [ ] Support variant sampling rate
