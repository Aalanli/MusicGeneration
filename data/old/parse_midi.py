from pathlib import Path

import py_midicsv as pm

# TODO organize


# 60000 / (120 * 480) = microseconds per tick
# 960 ticks per second for 480 ticks per quarter
tps_480 = 960
# 768 ticks per second for 384 ticks per quarter
tps_384 = 768

damper = 64
damper_on = 64
damper_off = 63

sostenuto = 66

time_bins = 50
velocity_bins = 16

max_class_v2 = time_bins + velocity_bins + 177


def save_from_csv(csv, file_name):
    # Parse the CSV output of the previous command back into a MIDI file
    midi_object = pm.csv_to_midi(csv)

    # Save the parsed MIDI file to disk
    with open(file_name, "wb") as output_file:
        midi_writer = pm.FileWriter(output_file)
        midi_writer.write(midi_object)


def apply_bin(x, maximum, bins, minimum=0):
    if x > maximum:
        x = maximum
    bin_size = (maximum - minimum) / bins
    return int(round((x - minimum) / bin_size, 0))


def inverse_bin(x, maximum, bins, minimum=0):
    bin_size = (maximum - minimum) / bins
    return x * bin_size + minimum


def format_head(ticks=480):
    # standard header for midi
    standard = [
        f'0, 0, Header, 1, 2, {ticks}',
        '1, 0, Start_track',
        '1, 0, Tempo, 500000',
        '1, 0, Time_signature, 4, 2, 24, 8',
        '1, 1, End_track',
        '2, 0, Start_track',
        '2, 0, Program_c, 0, 0'
    ]
    return standard


def format_tail(last_time):
    # standard close
    standard = [
        f'2, {last_time + 1}, End_track',
    ]
    return standard


def format_note_line(time, note, velocity=60):
    return f'2, {time}, Note_on_c, 0, {note}, {velocity}'


def to_midi_events(file):
    """
    returns [control type, time, value, velocity]
    """
    csv = pm.midi_to_csv(file)
    events = [['ticks', int(csv[0].split(', ')[-1])]]
    for i in csv[1:]:
        x = i.split(', ')
        if 'Note_on_c' in x or 'Control_c' in x:
            events.append([x[2], int(x[1]), int(x[-2]), int(x[-1])])
    return events


def augument(events: list, time_shifts: list, note_transpositions: list):
    """
    time_shifts: A list describing the amount of time stretch to apply to a new sample
    ex: [1.2, 0.8]
    note_transpositions: A list describing the increment of note transposition for each new sample
    ex: [1, -1, 2]; up a minor 2nd, down a minor 2nd, up a major 2nd

    accepts lists in the format [control type, time, value, velocity]
    """
    all_events = [[] for _ in range(len(time_shifts) * len(note_transpositions))]
    nt = len(note_transpositions)
    for e in events:
        for i, t in enumerate(time_shifts):
            for j, n in enumerate(note_transpositions):
                if e[0] == 'ticks':
                    all_events[i * nt + j].append(e)
                elif e[0] == 'Note_on_c' and (21 <= e[2] + n <= 108):
                    all_events[i * nt + j].append([e[0], int(e[1] * t), e[2] + n, e[3]])
                else:
                    all_events[i * nt + j].append([e[0], int(e[1] * t), e[2], e[3]])

    return all_events



def encode_categorical(midi_events):
    """Returns individidually categorized and binned events
    accepts lists in the format [control type, time, value, velocity]

    220 classes, no note off events
    """
    
    ticks = midi_events[0][1]
    tps = int(1000 * 120 * ticks / 60000)  # ticks per second

    # ['time_shift': 0 - time_bins], 
    # ['set_note': 21 - 109], 
    # ['set_velocity': 0 - velocity bins]
    events = []
    note_state = {i: False for i in range(21, 110)}  # is note sustained by damper

    last_time = 0
    last_velocity = 1
    sustain = False

    def time_shift(t):
        nonlocal last_time
        t_shift = t - last_time
        if t_shift > tps:  # if time shift is greater than maximum bin
            tn = t_shift // tps
            t_shift = t_shift % tps
            for _ in range(tn):
                events.append(['time_shift', time_bins])
            last_time = t
        time_bin = apply_bin(t_shift, tps, time_bins)
        if time_bin != 0:
            events.append(['time_shift', time_bin])
            last_time = t


    for i in midi_events[1:]:
        # only care about the damper control
        if i[0] == 'Control_c' and i[2] == damper:
            if i[3] >= damper_on:
                sustain = True
            else:
                sustain = False
                if True in note_state.values():
                    time_shift(i[1])
                    if last_velocity != 0:
                        events.append(['set_velocity', 0])
                        last_velocity = 0
                    for k in note_state:
                        if note_state[k] is True:
                            events.append(['set_note', k])
                            note_state[k] = False

        if i[0] == 'Note_on_c':
            _, t, n, v = i

            time_shift(t)
        
            if v == 0:  # note is off
                if sustain:
                    note_state[n] = True
                elif last_velocity != 0:
                    events.append(['set_velocity', 0])
                    last_velocity = 0
            else:
                v = apply_bin(v, 127, 32, 1)
                if v != last_velocity:
                    events.append(['set_velocity', v])
                    last_velocity = v
            if not(sustain and v == 0):
                
                events.append(['set_note', n])

    return events


def encode_categorical_v2(midi_events):
    """Returns individidually categorized and binned events
    accepts lists in the format [control type, time, value, velocity]

    246 classes, with note off events
    """
    
    ticks = midi_events[0][1]
    tps = int(1000 * 120 * ticks / 60000)  # ticks per second

    # ['time_shift': 0 - time_bins], 
    # ['set_note': 21 - 109], 
    # ['set_velocity': 0 - velocity bins]
    events = []
    note_state = {i: False for i in range(21, 110)}  # is note sustained by damper
    sost_state = {i: False for i in range(21, 110)}  # is note sustained by sustenuto

    last_time = 0
    last_velocity = 1
    sustain = False
    sost = False

    def time_shift(t):
        nonlocal last_time
        t_shift = t - last_time
        
        while t_shift > tps:
            events.append(['time_shift', time_bins])
            t_shift -= tps
            last_time += tps

        local_t_bin = apply_bin(t_shift, tps, time_bins)
        if local_t_bin != 0:
            events.append(['time_shift', local_t_bin])
            last_time = t


    for i in midi_events[1:]:
        # damper control
        if i[0] == 'Control_c' and i[2] == damper:
            if i[3] >= damper_on:
                sustain = True
                if sost:
                    # damper carries sostenuto sustains
                    note_state = sost_state.copy()
            else:
                sustain = False
                if True in note_state.values():
                    if not sost_state == note_state:
                        # if sost and damper states are identical, nothing will change, so time does not shift
                        time_shift(i[1])
                    for k in note_state:
                        if note_state[k] is True:
                            if not sost_state[k]:
                                events.append(['note_off', k])
                            note_state[k] = False
        
        # sostenuto
        if i[0] == 'Control_c' and i[2] == sostenuto:
            if i[3] >= damper_on:
                sost = True
            else:
                sost = False
                if True in sost_state.values():
                    if not sustain:
                        # only time shift when note operations are certain
                        time_shift(i[1])
                    for k in sost_state:
                        if sost_state[k] is True:
                            if not sustain:
                                # do not turn off note when sustain is also on
                                events.append(['note_off', k])
                            note_state[k] = False

        if i[0] == 'Note_on_c':
            _, t, n, v = i
        
            if v == 0:  # note is off
                if not sost:
                    sost_state[n] = False

                if sustain:
                    note_state[n] = True
                elif not sost_state[n]:
                    time_shift(t)
                    events.append(['note_off', n])
            else:
                if not sost:
                    sost_state[n] = True

                time_shift(t)
                v = apply_bin(v, 127, velocity_bins - 1, 1) + 1
                if v != last_velocity:
                    events.append(['set_velocity', v])
                    last_velocity = v
                                
                events.append(['set_note', n])

    return events


def decode_categorical(events, ticks):
    csv = format_head(ticks)
    tps = int(1000 * 120 * ticks / 60000)

    absolute_time = 0
    global_velocity = 1

    for x in events:
        e, d = x

        if e == 'time_shift':
            absolute_time += inverse_bin(d, tps, time_bins)
        if e == 'set_velocity':
            if d == 0:
                global_velocity = 0
            else:
                global_velocity = inverse_bin(d - 1, 127, velocity_bins - 1, 1)
        if e == 'set_note':
            csv.append(format_note_line(int(round(absolute_time, 0)), d, int(global_velocity)))
    
    csv.extend(format_tail(int(absolute_time)))
    
    return csv


def decode_categorical_v2(events, ticks):
    csv = format_head(ticks)
    tps = int(1000 * 120 * ticks / 60000)

    absolute_time = 0
    global_velocity = 1

    for x in events:
        e, d = x

        if e == 'time_shift':
            absolute_time += inverse_bin(d, tps, time_bins)
        if e == 'set_velocity':
            global_velocity = inverse_bin(d - 1, 127, velocity_bins - 1, 1)
        if e == 'set_note':
            csv.append(format_note_line(int(round(absolute_time, 0)), d, int(global_velocity)))
        if e == 'note_off':
            csv.append(format_note_line(int(round(absolute_time, 0)), d, 0))
    
    csv.extend(format_tail(int(absolute_time)))
    
    return csv


def encode_categorical_classes(events):
    """
    velocities = 0 - (velocity_bins - 1)
    times = velocity_bins - (velocity_bins + time_bins - 1) (+ velocity_bins - 1)
    notes = (velocity_bins + time_bins) - ... (+ velocity_bins + time_bins - 21)
    """
    categories = []
    for i in events:
        if i[0] == 'set_velocity':
            categories.append(i[1])
        if i[0] == 'time_shift':
            categories.append(i[1] + velocity_bins - 1)
        if i[0] == 'set_note':
            categories.append(i[1] + velocity_bins + time_bins - 21)
    
    return categories


def encode_categorical_classes_v2(events):
    """
    velocities = 0 - velocity_bins
    times = (velocity_bins + 1) - (velocity_bins + time_bins + 1); (+ velocity_bins + 1)
    notes = (time_bins + velocity_bins + 2) - (prev + 87) (+ time_bins + velocity_bins - 19)
    note_off = (time_bins + velocity_bins + 90) - (prev + 87) (+ time_bins + velocity_bins + 69)
    """
    categories = []
    for i in events:
        if i[0] == 'set_velocity':
            categories.append(i[1])
        if i[0] == 'time_shift':
            categories.append(i[1] + velocity_bins + 1)
        if i[0] == 'set_note':
            categories.append(i[1] + time_bins + velocity_bins - 19)
        if i[0] == 'note_off':
            categories.append(i[1] + time_bins + velocity_bins + 69)
    
    return categories


def decode_categorical_classes(classes):
    events = []
    for i in classes:
        if 0 <= i <= velocity_bins:
            events.append(['set_velocity', i])
        if velocity_bins + 1 <= i <= 131:
            events.append(['time_shift', i - 31])
        if 132 <= i <= 219:
            events.append(['set_note', i - 111])
    return events


def decode_categorical_classes_v2(classes):
    events = []
    for i in classes:
        if 0 <= i <= velocity_bins:
            events.append(['set_velocity', i])
        if velocity_bins + 1 <= i <= velocity_bins + time_bins + 1:
            events.append(['time_shift', i - velocity_bins - 1])
        if time_bins + velocity_bins + 2 <= i <= time_bins + velocity_bins + 89:
            events.append(['set_note', i - time_bins - velocity_bins + 19])
        if i >= time_bins + velocity_bins + 90:
            events.append(['note_off', i - time_bins - velocity_bins - 69])
    return events
