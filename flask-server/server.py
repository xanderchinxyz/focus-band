# -*- coding: utf-8 -*-
"""
Estimate Relaxation from Band Powers
This example shows how to buffer, epoch, and transform EEG data from a single
electrode into values for each of the classic frequencies (e.g. alpha, beta, theta)
Furthermore, it shows how ratios of the band powers can be used to estimate
mental state for neurofeedback.
The neurofeedback protocols described here are inspired by
*Neurofeedback: A Comprehensive Review on System Design, Methodology and Clinical Applications* by Marzbani et. al
Adapted from https://github.com/NeuroTechX/bci-workshop
"""

from flask import Flask, jsonify, render_template
import json

import numpy as np  # Module that simplifies computations on matrices
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import utils  # Our own utility functions
import cv2
import time

############## FLASK APP ##############
app = Flask(__name__)

################### EEG EXTRACTION STUFF #######################

""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 5
# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1
# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8
# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
# Index of the channel(s) (electrodes) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL = [0]

notRanOnce = True
dataPoints = []

@app.route("/reset")
def home():
    global notRanOnce
    notRanOnce = True
    return


@app.route("/get", methods=['GET'])
def getEEG():
    overallCount = 0
    focusCount = 0
    dataPoints.clear()
    global notRanOnce
    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=10)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    description = info.desc()

    # Get the sampling frequency
    # This is an important value that represents how many EEG data points are
    # collected in a second. This influences our frequency band calculation.
    # for the Muse 2016, this should always be 256
    fs = int(info.nominal_srate())

    """ 2. INITIALIZE BUFFERS """

    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, 128))

    """ 3. GET DATA """

    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print('Press Ctrl-C in the console to break the while loop.')
    startTime = time.time()
    try:
        # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
        while (time.time() - startTime) < 16:
        #while True:    
            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=int(SHIFT_LENGTH * fs))

            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

            # Update EEG buffer with the new data
            eeg_buffer, filter_state = utils.update_buffer(
                eeg_buffer, ch_data, notch=True,
                filter_state=filter_state)

            """ 3.2 COMPUTE BAND POWERS """
            # Get newest samples from the buffer
            data_epoch = utils.get_last_data(eeg_buffer,
                                             EPOCH_LENGTH * fs)

            # Compute band powers
            band_powers = utils.compute_band_powers(data_epoch, fs)

            band_buffer, _ = utils.update_buffer(band_buffer,
                                                 np.asarray([band_powers]))
            # Compute the average band powers for all epochs in buffer
            # This helps to smooth out noise
            smooth_band_powers = np.mean(band_buffer, axis=0)
            #print("h: ", smooth_band_powers)
            #print(smooth_band_powers)

            # print('Delta: ', band_powers[Band.Delta], ' Theta: ', band_powers[Band.Theta],
            #       ' Alpha: ', band_powers[Band.Alpha], ' Beta: ', band_powers[Band.Beta])

            """ 3.3 COMPUTE NEUROFEEDBACK METRICS """
            # These metrics could also be used to drive brain-computer interfaces

            # Alpha Protocol:
            # Simple redout of alpha power, divided by delta waves in order to rule out noise
            #alpha_metric = smooth_band_powers[Band.Alpha]/smooth_band_powers[Band.Delta]
            #alpha_metric = smooth_band_powers[Band.Alpha]
            #print('Alpha Relaxation: ', alpha_metric)

            # Beta Protocol:
            # Beta waves have been used as a measure of mental activity and concentration
            # This beta over theta ratio is commonly used as neurofeedback for ADHD
            #beta_metric = smooth_band_powers[Band.Beta]/smooth_band_powers[Band.Theta]
            #beta_metric = smooth_band_powers[Band.Beta]
            #print('Beta Concentration: ', beta_metric)

            # Alpha/Theta Protocol:
            # This is another popular neurofeedback metric for stress reduction
            # Higher theta over alpha is supposedly associated with reduced anxiety
            #theta_metric = smooth_band_powers[Band.Theta]/smooth_band_powers[Band.Alpha]
            #theta_metric = smooth_band_powers[Band.Theta]
            #print('Theta Relaxation: ', theta_metric)

            deltaAvg = np.mean(smooth_band_powers[0:4])
            thetaAvg = np.mean(smooth_band_powers[4:8])
            alphaAvg = np.mean(smooth_band_powers[8:12])
            betaAvg = np.mean(smooth_band_powers[12:30])
            gammaAvg = np.mean(smooth_band_powers[30:40])
            width = 10

            dataPoints.append(alphaAvg/deltaAvg)
            # # Create a black image
            img = np.zeros((600, 1000, 3), dtype = np.uint8)
            for x in range(len(smooth_band_powers)):
                if x < 4:
                    cv2.rectangle(img, (x * width, 500), (width + x * width, int(500-smooth_band_powers[x]*10)), (255, 0, 0), -1)
                elif x >= 4 and x < 8:
                    cv2.rectangle(img, (x * width, 500), (width + x * width, int(500-smooth_band_powers[x]*10)), (0, 255, 0), -1)
                elif x >= 8 and x < 12:
                    cv2.rectangle(img, (x * width, 500), (width + x * width, int(500-smooth_band_powers[x]*10)), (0, 0, 255), -1)
                elif x >= 12 and x < 30:
                    cv2.rectangle(img, (x * width, 500), (width + x * width, int(500-smooth_band_powers[x]*10)), (0, 255, 255), -1)
                elif x >= 30 and x < 40:
                    cv2.rectangle(img, (x * width, 500), (width + x * width, int(500-smooth_band_powers[x]*10)), (255, 0, 255), -1)

            cv2.rectangle(img, (500, 500), (600, int(500-20*alphaAvg/deltaAvg)), (255, 0, 0), -1)
            cv2.rectangle(img, (600, 500), (700, int(500-20*betaAvg/thetaAvg)), (0, 255, 0), -1)
            cv2.rectangle(img, (700, 500), (800, int(500-20*thetaAvg/alphaAvg)), (0, 0, 255), -1)
            if(notRanOnce):
                cv2.imshow("Image", img)  # creates window
                cv2.setWindowProperty("Image", cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(1)

    except KeyboardInterrupt:
        print('Closing!')
    
    cv2.destroyWindow("Image")
    if(100*np.mean(dataPoints) > 100):
        percentage = 100
    else:
        percentage = 100*np.mean(dataPoints)
    
    if(notRanOnce):
        notRanOnce = False
        return ({
            "count": percentage,
            "points": list(dataPoints),
            "state": list(smooth_band_powers[:2])
        })


if __name__ == "__main__":
    app.run(debug=True)
    