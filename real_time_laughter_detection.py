from os import path
import torch
import numpy as np
import configs
import laugh_segmenter
from utils import audio_utils, data_loaders, torch_utils
from functools import partial
import pyaudio
import datetime
import time
from collections import deque

model_path = 'checkpoints/in_use/resnet_with_augmentation'
config = configs.CONFIG_MAP['resnet_with_augmentation']
threshold = 0.5
min_length = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

# Load the Model

model = config['model'](
    dropout_rate=0.0,
    linear_layer_size=config['linear_layer_size'],
    filter_sizes=config['filter_sizes']
)
feature_fn = config['feature_fn']
model.set_device(device)

if path.exists(model_path):
    torch_utils.load_checkpoint(
        model_path+'/best.pth.tar', model, map_location=device)
    model.eval()
else:
    raise Exception(f"Model checkpoint not found at {model_path}")

audio_system = pyaudio.PyAudio()  # start the PyAudio class

NUMBER_OF_FRAMES = 44
HOP_LENGTH = 186
CHUNK_SIZE = 1
# number of data points to read at a time (number of frames * hop length?)
CHUNK = CHUNK_SIZE*NUMBER_OF_FRAMES*HOP_LENGTH
RATE = 8000  # time resolution of the recording device (Hz)
# RATE = int(audio_system.get_device_info_by_index(0)['defaultSampleRate'])

max_number_of_audio_time_series = 4
audio_time_series_buffer = deque([])

print(f'Number of time series processed at a time: {max_number_of_audio_time_series}')

def buffer_time_series(in_data, frame_count, time_info, status):
    audio_time_series_buffer.append(np.frombuffer(in_data, dtype=np.float32))
    return (in_data, status)


def process_time_series():
    if (len(audio_time_series_buffer) > 2):
        audio_time_series_buffer.clear()
        print('Time series queue became too large (queue cleared). Adjust your parameters to avoid this issue.')
        return

    if (len(audio_time_series_buffer) < 1):
        time.sleep(0.1)
        return

    audio_time_series = np.array(audio_time_series_buffer.popleft(), dtype=np.float32).flatten()

    # Load features

    inference_dataset = data_loaders.RealTimeInferenceDataset(
        audio_time_series=audio_time_series,
        feature_fn=feature_fn,
        sr=RATE,
        n_frames=NUMBER_OF_FRAMES
    )

    collate_fn = partial(
        audio_utils.pad_sequences_with_labels,
        expand_channel_dim=config['expand_channel_dim']
    )

    inference_generator = torch.utils.data.DataLoader(
        inference_dataset,
        num_workers=0,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Make Predictions

    probabilities = []
    for model_inputs, _ in inference_generator:
        x = torch.from_numpy(model_inputs).float().to(device)
        predictions = model(x).cpu().detach().numpy().squeeze()
        if len(predictions.shape) == 0:
            predictions = [float(predictions)]
        else:
            predictions = list(predictions)
        probabilities += predictions
    probabilities = np.array(probabilities)

    #probs = laugh_segmenter.lowpass(probs)
    #instances = get_laughter_instances(probs, threshold=threshold)

    laughers = []
    for probability_index in range(len(probabilities)):
        probability = np.min(probabilities[probability_index:probability_index+1])
        if probability > threshold:
            laughers.append(probability)


    number_of_probabilities = len(probabilities)
    number_of_laughers = len(laughers)
    if number_of_laughers > 0:
        laugher_ratio = number_of_laughers/number_of_probabilities
        laugher_ratio_string = '{:.2f}%'.format(laugher_ratio*100).rjust(7)

        average_laugher_probability = np.average(laughers)
        average_laugher_probability_string = '{:.2f}%'.format(average_laugher_probability*100).rjust(7)
        average_laugher_probability_visual = ''
        for _ in range(int(average_laugher_probability * 100)):
            average_laugher_probability_visual += '|'
        average_laugher_probability_visual = average_laugher_probability_visual.ljust(100, '.')

        print(f'{datetime.datetime.now()} \t Ratio: {laugher_ratio_string} \t Average: {average_laugher_probability_string} {average_laugher_probability_visual}')
    else:
        print(f'{datetime.datetime.now()}')

stream = audio_system.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK,
    stream_callback=buffer_time_series
)  # uses default input device


while True:
    try:
        process_time_series()
    # ctrl-c
    except KeyboardInterrupt:
        stream.stop_stream()
        stream.close()
        # Release PortAudio system resources (6)
        audio_system.terminate()
        quit()