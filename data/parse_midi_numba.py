# %%
from dataclasses import dataclass
from typing import Tuple
import py_midicsv as pm
import numpy as np
import copy

from numba import njit

@dataclass(frozen=True)
class MidiParams:
    damper: int = 64    # id representing damper pedal
    sostenuto: int = 66 # id representing sostenuto pedal
    pedal_on: int = 64  # threshold for pedal to be on
    pedal_off: int = 63 # threshold for pedal to be off

@dataclass
class RawMidiStream:
    """
    Raw stream of midi events consisting of absolution time events
    where events.shape = [N, 4] where indices represent (control_id, abs_time, value, velocity)
    For control ids:
    'Note_on_c' = 1
    'Control_c' = 0
    """
    ticks: int
    events: np.ndarray


@dataclass
class MidiStream:
    """
    Represents a stream of midi events, where TimeShifts are relative time shifts between
    notes, and notes capture their own duration.
    events.shape = [N, 3] where indices represent (note, velocity, duration)
    A timeshift event is represented by [-1, 0, duration]
    """
    ticks: int
    events: np.ndarray


def parse_raw_midi_stream(file: str, dtype=np.int64) -> RawMidiStream:
    csv = pm.midi_to_csv(file)
    ticks = int(csv[0].split(', ')[-1])
    events = []
    for i in range(len(csv)):
        event = csv[i].split(', ')
        buffer = []
        if 'Note_on_c' == event[2] or 'Control_c' == event[2]:
            buffer.append(int(event[2] == 'Note_on_c'))
            buffer.append(int(event[1]))
            buffer.append(int(event[-2]))
            buffer.append(int(event[-1]))
            events.append(buffer)

    buffer = np.array(events, dtype=dtype)
    return RawMidiStream(ticks, buffer)

@njit(nogil=True)
def clean_raw_stream_(stream: np.ndarray, pedal_on=64) -> np.ndarray:
    a = np.empty_like(stream)
    a[0, :] = stream[0, :]
    length = 1
    damper_on = False
    sost_on = False
    for i in range(1, stream.shape[0]):
        if stream[i, 0] == 0: # if the control_id is "Control_c"
            if stream[i, 2] == 66 and ((stream[i, 3] >= pedal_on) != sost_on):
                a[length, :] = stream[i, :]
                sost_on = not sost_on
                length += 1
            elif stream[i, 2] == 64 and ((stream[i, 3] >= pedal_on) != damper_on):
                a[length, :] = stream[i, :]
                damper_on = not damper_on
                length += 1
        else:
            a[length, :] = stream[i, :]
            length += 1
    return a[:length]

@njit(nogil=True)
def raw_stream_to_stream_(events: np.ndarray, pedal_on=64) -> np.ndarray:
    events = clean_raw_stream_(events, pedal_on)
    new_events = np.empty((events.shape[0], 3), dtype=events.dtype)

    # array representing all the note states, True means pressed, False means not pressed
    note_pressed = np.zeros(110-21, dtype=np.bool8)


    # array representing if a note is on or note, different from note pressed, 
    # since a note can be still sound and not be pressed due to damper and sostenuto
    damper_buf = np.zeros_like(note_pressed)
    sost_buf = np.zeros_like(damper_buf)

    # array of the position in list 'events' that the last note (the index) was pressed
    # and the absolution time at which the note was pressed
    release_buf = np.zeros((110 - 21, 2), dtype=events.dtype)

    # global variables throughout the track
    time = 0
    damper_on = False

    idx = 0

    for i in range(0, events.shape[0]):
        control_type, abs_time, abs_value, velocity = events[i]
        if control_type == 1: # 'Note_on_c'
            value = abs_value - 21
            if velocity > 0: # a note is being pressed
                # shift the note state timing by a relative timing
                # (note, velocity, duration)
                if abs_time > time:
                    new_events[idx, 0] = -1
                    new_events[idx, 1] = 0
                    new_events[idx, 2] = abs_time - time
                    time = abs_time
                    idx += 1
                
                if new_events[release_buf[value, 0], 2] == -1:
                    # if the last note pressed did not end yet, this case happens when the 
                    # damper is still on, but a note gets pressed multiple times
                    new_events[release_buf[value, 0], 2] = abs_time - release_buf[value, 1]
                
                note_pressed[value] = True
                new_events[idx, 0] = value
                new_events[idx, 1] = velocity
                new_events[idx, 2] = -1
                idx += 1
                release_buf[value, 0] = idx - 1
                release_buf[value, 1] = abs_time
                if damper_on:
                    damper_buf[value] = True

            else: # a note is released
                note_pressed[value] = False
                if not damper_on and not sost_buf[value]:
                    # if neither damper is on or the note is still captured by sost
                    note_idx = release_buf[value, 0]
                    new_events[note_idx, 2] = abs_time - release_buf[value, 1]
        
        elif control_type == 0: # control_id is 'Control_c'
            if abs_value == 64: # damper
                if velocity >= pedal_on: # damper pedal is pressed
                    damper_on = True
                    damper_buf |= note_pressed
                else: # damper pedal is released
                    # turn off only notes that are not currently pressed
                    # and not sustained by sost
                    damper_buf &= np.invert(note_pressed) & np.invert(sost_buf)
                    # now damper_buf only consists of notes to turn off
                    for i in range(0, 110 - 21):
                        if damper_buf[i]:
                            note_idx = release_buf[i, 0]
                            new_events[note_idx, 2] = abs_time - release_buf[i, 1]
                    damper_buf &= False
                    damper_on = False

            elif abs_value == 66: # sostenuto
                if velocity >= pedal_on: # sostenuto pedal is pressed
                    # sostenuto pedal captures all notes that are pressed
                    sost_buf |= note_pressed
                else: # sostenuto pedal is released and damper is not on
                    if not damper_on:
                        # release every note captured by sostenuto that is not currently pressed
                        sost_buf &= np.invert(note_pressed)
                        for i in range(0, 110 - 21):
                            if sost_buf[i]:
                                note_idx = release_buf[i, 0]
                                new_events[note_idx, 2] = abs_time - release_buf[i, 1]
                    else:
                        # if damper is on, pass notes held by sostenuto to damper
                        damper_buf |= sost_buf
                    # every note captured by sostenuto is lost anyways, if the pedal is released
                    sost_buf &= False
    
    # in case pedal never lifts before track finishes
    for i in range(0, 110 - 21):
        if sost_buf[i] or damper_buf[i]:
            note_idx = release_buf[i, 0]
            if new_events[note_idx, 2] != -1:
                continue
            new_events[note_idx, 2] = abs_time - release_buf[i, 1] + 1

    return new_events[:idx]


def raw_stream_to_stream(stream: RawMidiStream, pedal_on=64) -> MidiStream:
    new_events = raw_stream_to_stream_(stream.events, pedal_on)
    return MidiStream(stream.ticks, new_events)

@njit(nogil=True)
def stream_to_raw_stream_(events: np.ndarray) -> np.ndarray:
    elems = np.sum(events[:, 0] != -1) * 2
    new_events = np.empty((elems, 4), dtype=events.dtype)
    abs_time = 0
    idx = 0

    for i in range(0, events.shape[0]):
        if events[i, 0] == -1: # shift time id
            abs_time += events[i, 2] # shift by duration
        else:
            new_events[idx] = [1, abs_time, events[i, 0] + 21, events[i, 1]]
            new_events[idx + 1] = [1, abs_time + events[i, 2], events[i, 0] + 21, 0]
            idx += 2
    
    # quite strange, apparently, even though no note has a duration of 0, 
    # the order of notes with the same abs_time still matters, so a stable sort is
    # required
    order = np.argsort(new_events[:, 1], kind='mergesort')
    new_events = new_events[order]
    return new_events


def stream_to_raw_stream(stream: MidiStream) -> RawMidiStream:
    return RawMidiStream(stream.ticks, stream_to_raw_stream_(stream.events))


def raw_stream_to_midi_file(file_name: str, stream: RawMidiStream) -> None:
    csv = [
        f'0, 0, Header, 1, 2, {stream.ticks}',
        '1, 0, Start_track',
        '1, 0, Tempo, 500000',
        '1, 0, Time_signature, 4, 2, 24, 8',
        '1, 1, End_track',
        '2, 0, Start_track',
        '2, 0, Program_c, 0, 0'
    ]
    # everything has been compressed to note on and note off commands after 
    # stream_to_raw_stream(to_midi_stream(x))
    for event in stream.events:
        if event[0] == 0:
            control_type = 'Control_c'
        else:
            control_type = 'Note_on_c'
        csv.append(
            f'2, {event[1]}, {control_type}, 0, {event[2]}, {event[3]}'
        )
    
    # append tail to end the track
    csv.append(
        f'2, {stream.events[-1, 1] + 1}, End_track'
    )
    
    midi_obj = pm.csv_to_midi(csv)

    with open(file_name, 'wb') as f:
        midi_writer = pm.FileWriter(f)
        midi_writer.write(midi_obj)


def compute_lin_bin(lo, hi, bins):
    return np.full(bins, (hi - lo) / bins, dtype=np.float64)

def compute_duration_bin_arr_piecewise2(lin_bins, exp_bins):
    lo = 0.0020833333333333333
    mi0 = 0.08133958333333334
    mi1 = 0.5304583333333334
    mi2 = 5.3915083333333325
    hi = 105.67708333333333
    
    lin_arr0 = compute_lin_bin(lo, mi0, int(lin_bins * 0.2))
    lin_bins -= lin_arr0.shape[0]
    lin_arr1 = compute_lin_bin(lo, mi1, int(lin_bins * 0.4))
    lin_bins -= lin_arr1.shape[0]
    lin_arr2 = compute_lin_bin(mi1, mi2, lin_bins)

    exp_coef = (hi - mi2) / (np.exp2(exp_bins + 1) - 2)
    exp_arr = exp_coef * np.exp2(np.arange(1, exp_bins + 1, 1), dtype=np.float64)
    return np.cumsum(np.concatenate([lin_arr0, lin_arr1, lin_arr2, exp_arr]))


@dataclass(frozen=True)
class BinParams:
    note_bins: int = 89
    velocity_bins: int = 127


def bin_stream(stream: np.ndarray, bin_arr: np.ndarray) -> np.ndarray:
    """
    stream has dtype float64
    stream is an array of shape [N, 3], where the last dim represents (note, velocity, duration)
    notes are already integers from -1 to 87, where -1 represent a time shift
    velocites are already integers from 0 to 126
    need to bin durations according to bin_arr

    binning procedure bins the three streams
    returns: [N, (binned note, binned velocity, binned duration)]
    """
    new_stream = stream.astype(np.int32)
    new_stream[:, 0] += 1
    new_stream[:, 2] = np.digitize(stream[:, 2], bin_arr).astype(np.int32)
    new_stream[np.where(new_stream[:, 2] == bin_arr.shape[0]), 2] -= 1
    return new_stream


def unbin_stream(stream: np.ndarray, bin_arr: np.ndarray) -> np.ndarray:
    stream = copy.deepcopy(stream)
    stream[:, 0] -= 1
    stream[:, 2] = bin_arr[stream[:, 2]]
    return stream


