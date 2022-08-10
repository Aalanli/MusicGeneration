
import _pickle as pickle
import tqdm

import ray

from data.parse_midi import *


def give_file_paths(data_dir):
    data_dir = Path(data_dir)
    file_paths = sorted(data_dir.glob('**/*.midi'))
    file_paths.extend(sorted(data_dir.glob('**/*.mid')))
    return [str(i) for i in file_paths]


class WriteNpArray:
    def __init__(self, time_shifts, note_transpositions) -> None:
        self.time_shifts = time_shifts
        self.note_transpositions = note_transpositions
    
    @ray.remote
    def format_categorical(self, file, save_name):
        x = to_midi_events(file)
        x = augument(x, time_shifts=self.time_shifts, note_transpositions=self.note_transpositions)

        for i, data in enumerate(x):
            #x[i] = encode_categorical_classes(encode_categorical(x[i]))
            d = encode_categorical_classes_v2(encode_categorical_v2(data))

            with open(save_name + f'_{i}.pkl', 'wb') as f:
                pickle.dump(d, f)


    def write_to_array(self, file_paths, save_directory, bins=25):
        ray.init()
        pbar = tqdm(total=len(file_paths))
        length = len(file_paths)
        index = 0
        while index < length:
            if index + bins < length:
                examples = [self.format_categorical.remote(self, file, save_directory + f'/{i + index}') for i, file in enumerate(file_paths[index:index + bins])]
            else:
                examples = [self.format_categorical.remote(self, file, save_directory + f'/{i + index}') for i, file in enumerate(file_paths[index:])]
            for _ in ray.get(examples):
                pbar.update(1)
                index += 1
        pbar.close()
        ray.shutdown()
