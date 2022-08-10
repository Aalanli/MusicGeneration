# %%
from dataclasses import dataclass

from typing import List, Union
import py_midicsv as pm
import numpy as np

@dataclass(frozen=True)
class MidiParams:
    damper: int = 64    # id representing damper pedal
    sostenuto: int = 66 # id representing sostenuto pedal
    pedal_on: int = 64  # threshold for pedal to be on
    pedal_off: int = 63 # threshold for pedal to be off

midi_params = MidiParams()

@dataclass
class Note:
    note: int
    velocity: int
    duration: int

@dataclass
class TimeShift:
    duration: int

@dataclass
class MidiStream:
    """
    Represents a stream of midi events, where TimeShifts are relative time shifts between
    notes, and notes capture their own duration
    """
    ticks: int
    events: List[Union[Note, TimeShift]]

@dataclass
class RawMidiEvent:
    control_type: str
    abs_time: int
    value: int
    velocity: int

@dataclass
class RawMidiStream:
    ticks: int
    events: List[RawMidiEvent]


def parse_raw_midi_stream(file: str) -> RawMidiStream:
    """
        opens a midi file, and parses contents to a RawMidiStream, which
        contains all the necessary contents of the parsed file.
    """
    csv = pm.midi_to_csv(file)
    ticks = int(csv[0].split(', ')[-1])
    events = []
    for event_ in csv[1:]:
        event = event_.split(', ')
        if 'Note_on_c' in event or 'Control_c' in event:
            events.append(RawMidiEvent(event[2], int(event[1]), int(event[-2]), int(event[-1])))

    return RawMidiStream(ticks, events)


def join_adjacent_timeshifts(events: List[MidiStream]) -> List[MidiStream]:
    new_events = [events[0]]
    for event in events[1:]:
        if isinstance(event, TimeShift) and isinstance(new_events[-1], TimeShift):
            new_events[-1].duration += event.duration
        else:
            new_events.append(event)
    return new_events


def clean_raw_stream(stream: RawMidiStream) -> RawMidiStream:
    new_event = [stream.events[0]]
    damper_on = False
    sost_on = False
    for event in stream.events[1:]:
        if event.control_type == "Control_c":
            # if the Control_c event actually changes anything, append to the end of the list
            if event.value == midi_params.sostenuto and ((event.velocity >= midi_params.pedal_on) != sost_on):
                new_event.append(event)
                sost_on = not sost_on
            elif event.value == midi_params.damper and ((event.velocity >= midi_params.pedal_on) != damper_on):
                new_event.append(event)
                damper_on = not damper_on
        else:
            # otherwise its a note event, so always append
            new_event.append(event)
    return RawMidiStream(stream.ticks, new_event)


def raw_stream_to_stream(stream: RawMidiStream) -> MidiStream:
    """
        parses a RawMidiStream into a better representation,
        namely, to a representation that is not temporally dependent.
    """
    events = []

    # array representing all the note states, True means pressed, False means not pressed
    note_pressed = np.zeros([110-21], dtype=np.bool8)


    # array representing if a note is on or note, different from note pressed, 
    # since a note can be still sound and not be pressed due to damper and sostenuto
    damper_buf = np.zeros_like(note_pressed)
    sost_buf = np.zeros_like(damper_buf)

    # array of the position in list 'events' that the last note (the index) was pressed
    # and the absolution time at which the note was pressed
    release_buf = np.zeros([110 - 21, 2], dtype=np.int64)

    # global variables throughout the track
    time = 0
    damper_on = False

    for event in stream.events:
        if event.control_type == 'Note_on_c':
            value = event.value - 21
            if event.velocity > 0: # a note is being pressed
                # shift the note state timing by a relative timing
                if event.abs_time > time:
                    events.append(TimeShift(event.abs_time - time))
                    time = event.abs_time
                
                if events[release_buf[value, 0]].duration == -1:
                    # if the last note pressed did not end yet, this case happens when the 
                    # damper is still on, but a note gets pressed multiple times
                    events[release_buf[value, 0]].duration = event.abs_time - release_buf[value, 1]
                
                note_pressed[value] = True
                events.append(Note(event.value - 21, event.velocity, -1))
                release_buf[value, 0] = len(events) - 1
                release_buf[value, 1] = event.abs_time
                if damper_on:
                    damper_buf[value] = True

            else: # a note is released
                note_pressed[value] = False
                if not damper_on and not sost_buf[value]:
                    # if neither damper is on or the note is still captured by sost
                    note_idx = release_buf[value, 0]
                    events[note_idx].duration = event.abs_time - release_buf[value, 1]
        
        elif event.control_type == 'Control_c':
            if event.value == midi_params.damper:
                if event.velocity >= midi_params.pedal_on: # damper pedal is pressed
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
                            events[note_idx].duration = event.abs_time - release_buf[i, 1]
                    damper_buf &= False
                    damper_on = False

            elif event.value == midi_params.sostenuto:
                if event.velocity >= midi_params.pedal_on: # sostenuto pedal is pressed
                    # sostenuto pedal captures all notes that are pressed
                    sost_buf |= note_pressed
                else: # sostenuto pedal is released and damper is not on
                    if not damper_on:
                        # release every note captured by sostenuto that is not currently pressed
                        sost_buf &= np.invert(note_pressed)
                        for i in range(0, 110 - 21):
                            if sost_buf[i]:
                                note_idx = release_buf[i, 0]
                                events[note_idx].duration = event.abs_time - release_buf[i, 1]
                    else:
                        # if damper is on, pass notes held by sostenuto to damper
                        damper_buf |= sost_buf
                    # every note captured by sostenuto is lost anyways, if the pedal is released
                    sost_buf &= False
    
    # in case pedal never lifts before track finishes
    for i in range(0, 110 - 21):
        if sost_buf[i] or damper_buf[i]:
            note_idx = release_buf[i, 0]
            assert isinstance(events[note_idx], Note)
            if events[note_idx].duration != -1:
                continue
            events[note_idx].duration = event.abs_time - release_buf[i, 1] + 1

    return MidiStream(stream.ticks, events)


def stream_to_raw_stream(stream: MidiStream) -> RawMidiStream:
    events = []
    abs_time = 0
    # the relative timing offsets for     
    for event in stream.events:
        if isinstance(event, TimeShift):
            abs_time += event.duration
        else:
            # turn on the note
            events.append(RawMidiEvent('Note_on_c', abs_time, event.note + 21, event.velocity))
            # turn off the note
            events.append(RawMidiEvent('Note_on_c', abs_time + event.duration, event.note + 21, 0))
    
    events.sort(key=lambda ev: ev.abs_time)
    return RawMidiStream(stream.ticks, events)


def raw_stream_to_midi_file(file_name: str, stream: RawMidiStream) -> None:
    # header used to format the beginning of every midi track
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
        csv.append(
            f'2, {event.abs_time}, {event.control_type}, 0, {event.value}, {event.velocity}'
        )
    
    # append tail to end the track
    csv.append(
        f'2, {stream.events[-1].abs_time + 1}, End_track'
    )

    
    midi_obj = pm.csv_to_midi(csv)

    with open(file_name, 'wb') as f:
        midi_writer = pm.FileWriter(f)
        midi_writer.write(midi_obj)
