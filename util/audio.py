from __future__ import absolute_import, print_function

import scipy.io.wavfile as wav
import sys

try:
    from deepspeech.utils import audioToInputVector
except ImportError:
    import numpy as np
    from python_speech_features import mfcc
    from six.moves import range

    class DeprecationWarning:
        displayed = False

    def audioToInputVector(audio, fs, numcep, numcontext):
        if DeprecationWarning.displayed is not True:
            DeprecationWarning.displayed = True
            print('------------------------------------------------------------------------', file=sys.stderr)
            print('WARNING: libdeepspeech failed to load, resorting to deprecated code',      file=sys.stderr)
            print('         Refer to README.md for instructions on installing libdeepspeech', file=sys.stderr)
            print('------------------------------------------------------------------------', file=sys.stderr)

        # Get mfcc coefficients
        features = mfcc(audio, samplerate=fs, numcep=numcep)

        # Whiten inputs (TODO: Should we whiten?)
        features = (features - np.mean(features))/np.std(features)

        # Return results
        return features


def audiofile_to_input_vector(audio_filename, numcep, numcontext):
    r"""
    Given a WAV audio file at ``audio_filename``, calculates ``numcep`` MFCC features
    at every 0.01s time step with a window length of 0.025s. Appends ``numcontext``
    context frames to the left and right of each time step, and returns this data
    in a numpy array.
    """
    # Load wav files
    fs, audio = wav.read(audio_filename)

    return audioToInputVector(audio, fs, numcep, numcontext)
