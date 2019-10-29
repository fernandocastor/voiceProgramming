# The Ultimate Guide to Speech Recognition with Python by David Amos

The system for Voice Programming we want to build can be seen as a voice assistant (along the lines of Siri or Cortana). Very early work on this: Shoebox by IBM (1962) and Harpy at CMU (1976). 

Python already has an API for speech recognition, aptly-named ``speechrecognition``. To install it, one must have ``pyAudio`` installed. To install the latter, ``portaudio`` must have been installed. ``Portaudio`` is not Python-based and must be installed using some other tool such as MacOS's ``brew`` or ``apt-get``. In Python code, the name of the package (to be imported) is ``speech_recognition`` (with an underscore separating the two words). There are alternatives to ``speechrecognition``, although it seems to be the easiest to use (a list is available here [here](https://realpython.com/python-speech-recognition/)): 

- ``apiai``
- ``assemblyai``
- ``google-cloud-speech``
- ``pocketsphinx``
- ``watson-developer-cloud``
- ``wit``

The ``Recognizer`` class of the ``speechrecognition`` library can use a number of different methods for recognizing speech (Google, Bing, IBM Speech to Text, CMU Sphinx, among others). CMU Sphinx (method ``recognize_sphinx``) is the only one to work offline. However, it requires the installation of an additional package, ``pocketsphinx``. The latter library has a dependency that must be installed first, via ``brew``, called ``swig``. However, using ``pip`` to install ``pocketsphinx`` still did not work. Therefore, had to follow these instructions (from [here](https://github.com/bambocher/pocketsphinx-python/issues/28), transcripted literally): 

```
1. git clone --recursive https://github.com/bambocher/pocketsphinx-python
2. cd pocketsphinx-python
3. Edit file pocketsphinx-python/deps/sphinxbase/src/libsphinxad/ad_openal.c
4. Change

#include <al.h>
#include <alc.h>

to

#include <OpenAL/al.h>
#include <OpenAL/alc.h>

python setup.py install
```

In my tests using the Microphone, the Google API performed better than CMU Sphinx. Also tested [Wit.ai](http://wit.ai). The result was a bit better than Sphinx but worse than Google. Wit.ai is free, however, unlike Google.

To process speech, it is necessary to create a ``Recognizer``object. This is the object that will be responsible for processing the audio and turning it into text. The audio must come from one of two sources, either an audio file or the microphone. Audio files are useful for transcriptions. Multiple formats are accepted and parameters such as offset and duration can be set for recognition.

The ``Recognizer`` class makes available a method to deal with noise during speech recognition, ``adjust_for_ambient_noise()``. This method reads the first second (by default -- this can be set by a parameter named ``duration``) of the audio source and tries to calibrate the recognizer to the noise level of the audio.

To use the ``Recognizer`` with a microphone, one just needs to create a ``Microphone`` object and then invoke method ``listen()`` from the ``Recognizer`` class passing the ``Microphone`` object as argument.  This will return a source (similar to reading an audio file) that can be passed to one of the ``recognize_*()`` methods to perform the actual recognition. If a system has multiple microphones, it is possible to specify which one to use.

For input by microphone or audio file, unrecognizable input will result in an ``UnknownValueError`` exception being thrown.

Reference for the ``speechrecognition`` library: https://github.com/Uberi/speech_recognition/blob/master/reference/library-reference.rst