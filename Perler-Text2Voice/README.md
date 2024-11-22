---
library_name: transformers
tags:
- text-to-speech
- annotation
license: apache-2.0
language:
- en
pipeline_tag: text-to-speech
inference: false
datasets:
- parler-tts/mls_eng
- parler-tts/libritts_r_filtered
- parler-tts/libritts-r-filtered-speaker-descriptions
- parler-tts/mls-eng-speaker-descriptions
---

<img src="https://huggingface.co/datasets/parler-tts/images/resolve/main/thumbnail.png" alt="Parler Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>


# Parler-TTS Large v1

<a target="_blank" href="https://huggingface.co/spaces/parler-tts/parler_tts">
  <img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg" alt="Open in HuggingFace"/>
</a>

**Parler-TTS Large v1** is a 2.2B-parameters text-to-speech (TTS) model, trained on 45K hours of audio data, that can generate high-quality, natural sounding speech with features that can be controlled using a simple text prompt (e.g. gender, background noise, speaking rate, pitch and reverberation).

With [Parler-TTS Mini v1](https://huggingface.co/parler-tts/parler-tts-mini-v1), this is the second set of models published as part of the [Parler-TTS](https://github.com/huggingface/parler-tts) project, which aims to provide the community with TTS training resources and dataset pre-processing code.

## 📖 Quick Index
* [👨‍💻 Installation](#👨‍💻-installation)
* [🎲 Using a random voice](#🎲-random-voice)
* [🎯 Using a specific speaker](#🎯-using-a-specific-speaker)
* [Motivation](#motivation)
* [Optimizing inference](https://github.com/huggingface/parler-tts/blob/main/INFERENCE.md)

## 🛠️ Usage

### 👨‍💻 Installation

Using Parler-TTS is as simple as "bonjour". Simply install the library once:

```sh
pip install git+https://github.com/huggingface/parler-tts.git
```

### 🎲 Random voice


**Parler-TTS** has been trained to generate speech with features that can be controlled with a simple text prompt, for example:

```py
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-large-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-large-v1")

prompt = "Hey, how are you doing today?"
description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
```

### 🎯 Using a specific speaker

To ensure speaker consistency across generations, this checkpoint was also trained on 34 speakers, characterized by name (e.g. Jon, Lea, Gary, Jenna, Mike, Laura).

To take advantage of this, simply adapt your text description to specify which speaker to use: `Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise.`

```py
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-large-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-large-v1")

prompt = "Hey, how are you doing today?"
description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
```

**Tips**:
* We've set up an [inference guide](https://github.com/huggingface/parler-tts/blob/main/INFERENCE.md) to make generation faster. Think SDPA, torch.compile, batching and streaming!
* Include the term "very clear audio" to generate the highest quality audio, and "very noisy audio" for high levels of background noise
* Punctuation can be used to control the prosody of the generations, e.g. use commas to add small breaks in speech
* The remaining speech features (gender, speaking rate, pitch and reverberation) can be controlled directly through the prompt

## Motivation

Parler-TTS is a reproduction of work from the paper [Natural language guidance of high-fidelity text-to-speech with synthetic annotations](https://www.text-description-to-speech.com) by Dan Lyth and Simon King, from Stability AI and Edinburgh University respectively. 

Contrarily to other TTS models, Parler-TTS is a **fully open-source** release. All of the datasets, pre-processing, training code and weights are released publicly under permissive license, enabling the community to build on our work and develop their own powerful TTS models.
Parler-TTS was released alongside:
* [The Parler-TTS repository](https://github.com/huggingface/parler-tts) - you can train and fine-tuned your own version of the model.
* [The Data-Speech repository](https://github.com/huggingface/dataspeech) - a suite of utility scripts designed to annotate speech datasets.
* [The Parler-TTS organization](https://huggingface.co/parler-tts) - where you can find the annotated datasets as well as the future checkpoints.


## License

This model is permissively licensed under the Apache 2.0 license.
