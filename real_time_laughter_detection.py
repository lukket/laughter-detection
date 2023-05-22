# Example usage:
# python segment_laughter.py --input_audio_file=tst_wave.wav --output_dir=./tst_wave --save_to_textgrid=False --save_to_audio_files=True --min_length=0.2 --threshold=0.5

from os import path
import torch
import numpy as np
import configs
from utils import audio_utils, data_loaders, torch_utils
from functools import partial
import pyaudio
import datetime

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

MIN_CHUNK = 44*186  # number of data points to read at a time

CHUNK = 1*MIN_CHUNK  # number of data points to read at a time
# RATE = 8000  # time resolution of the recording device (Hz)
RATE = int(audio_system.get_device_info_by_index(0)['defaultSampleRate'])
print(f'Rate: {RATE}')


def callback(in_data, frame_count=None, time_info=None, status=None):
    data = np.frombuffer(in_data, dtype=np.float32)

    # Load features

    inference_dataset = data_loaders.RealTimeInferenceDataset(
        audio_time_series=data, feature_fn=feature_fn, sr=RATE)

    collate_fn = partial(audio_utils.pad_sequences_with_labels,
                         expand_channel_dim=config['expand_channel_dim'])

    inference_generator = torch.utils.data.DataLoader(
        inference_dataset, num_workers=0, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # Make Predictions

    probs = []
    for model_inputs, _ in inference_generator:
        x = torch.from_numpy(model_inputs).float().to(device)
        preds = model(x).cpu().detach().numpy().squeeze()
        if len(preds.shape) == 0:
            preds = [float(preds)]
        else:
            preds = list(preds)
        probs += preds
    probs = np.array(probs)

    # probs = laugh_segmenter.lowpass(probs)
    instances = get_laughter_instances(probs, threshold=threshold)

    return (in_data, status)


def get_laughter_instances(probs, threshold=0.5, min_length=0.2, fps=100.) -> list:
    instances = []
    current_list = []
    for i in range(len(probs)):
        probability = np.min(probs[i:i+1])
        percentage = int(probability * 100)
        # print(f'Probability: {probability}')
        timestamp = datetime.datetime.now()
        if probability > threshold:
            print(f'{timestamp} {percentage}%: 🤣')
            current_list.append(i)
        else:
            print(f'{timestamp} {percentage}%: .')
            if len(current_list) > 0:
                instances.append(current_list)
                current_list = []
    if len(current_list) > 0:
        instances.append(current_list)
    # instances = [frame_span_to_time_span(collapse_to_start_and_end_frame(i),fps=fps) for i in instances]
    # instances = [inst for inst in instances if inst[1]-inst[0] > min_length]
    return instances


stream = audio_system.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=RATE, input=True,
    frames_per_buffer=CHUNK,
    # stream_callback=callback
)  # uses default input device

# Wait for stream to finish (4)
while stream.is_active():
    try: 
        in_data = stream.read(CHUNK)
        callback(in_data=in_data)
    # ctrl-c
    except KeyboardInterrupt:
        exit()
    # any different error
    except RuntimeError as e:
        print(e)

# Close the stream (5)
stream.close()
# Release PortAudio system resources (6)
audio_system.terminate()