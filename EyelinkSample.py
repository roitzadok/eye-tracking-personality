import re
import os
import csv
from enum import Enum

SCREEN_CENTER_X = 962 #1920 / 2
SCREEN_CENTER_Y = 457 #1080 / 2
SCREEN_CENTER_R = 200 #70


class EyesState(Enum):
    BLINK = 0
    RIGHT = 1
    LEFT = 2
    BOTH = 3


class Fixation(object):
    def __init__(self, start_time: int, end_time: int):
        self.__start_time = start_time
        self.__end_time = end_time

    @property
    def duration(self):
        return self.__end_time - self.__start_time

    @property
    def start(self):
        return self.__start_time

    @property
    def end(self):
        return self.__end_time


class Frame(object):
    def __init__(self, frame_row: list):
        self.time = float(frame_row[0])
        self.left_eye_x = float(frame_row[1]) if frame_row[1] != '.' else None
        self.left_eye_y = float(frame_row[2]) if frame_row[2] != '.' else None
        self.left_eye_pupil_size = float(frame_row[3]) if frame_row[3] != '.' else None
        self.right_eye_x = float(frame_row[4]) if frame_row[4] != '.' else None
        self.right_eye_y = float(frame_row[5]) if frame_row[5] != '.' else None
        self.right_eye_pupil_size = float(frame_row[6]) if frame_row[6] != '.' else None

    def __str__(self):
        return f'{self.time}\t{self.return_value_or_dot(self.left_eye_x)}\t{self.return_value_or_dot(self.left_eye_y)}\t{self.return_value_or_dot(self.left_eye_pupil_size)}\t{self.return_value_or_dot(self.right_eye_x)}\t{self.return_value_or_dot(self.right_eye_y)}\t{self.return_value_or_dot(self.right_eye_pupil_size)}'

    @staticmethod
    def return_value_or_dot(value):
        return value if value else '.'

    @property
    def eyes_state(self) -> EyesState:
        if self.left_eye_x is not None and self.right_eye_x is not None:
            return EyesState.BOTH
        elif self.left_eye_x is not None:
            return EyesState.LEFT
        elif self.right_eye_x is not None:
            return EyesState.RIGHT
        else:
            return EyesState.BLINK

    @property
    def is_blink(self):
        return self.left_eye_x is None and self.right_eye_x is None

    @staticmethod
    def __is_point_inside_square(point_x, point_y, square_center_x, square_center_y, square_radius) -> bool:
        return (square_center_x - square_radius) <= point_x <= (square_center_x + square_radius) and (
                square_center_y - square_radius) <= point_y <= (square_center_y + square_radius)

    @staticmethod
    def __is_point_inside_circle(point_x, point_y, circle_center_x, circle_center_y, circle_radius):
        return (point_x - circle_center_x) ** 2 + (point_y - circle_center_y) ** 2 <= circle_radius ** 2

    def is_in_circle(self, center_x, center_y, r):
        eyes_state = self.eyes_state
        if eyes_state == EyesState.BLINK:
            return False

        elif eyes_state == EyesState.BOTH:
            return self.__is_point_inside_circle(self.left_eye_x, self.left_eye_y, center_x, center_y,
                                                 r) or self.__is_point_inside_circle(self.right_eye_x, self.right_eye_y,
                                                                                     center_x, center_y, r)
        elif eyes_state == EyesState.RIGHT:
            return self.__is_point_inside_circle(self.right_eye_x, self.right_eye_y, center_x, center_y, r)

        else:
            return self.__is_point_inside_circle(self.left_eye_x, self.left_eye_y, center_x, center_y, r)


class EyelinkSample(object):
    def __init__(self, file_path: str):
        self._file_path = file_path

    @property
    def file_path(self):
        return self._file_path

    @property
    def parsed(self) -> list:
        raise NotImplementedError


class AscEyelinkSample(EyelinkSample):
    def __init__(self, file_path: str):
        assert file_path.split('.')[-1] == 'asc', 'file type should be asc'
        super().__init__(file_path)
        self.__csv_files = None

    def __split_asc_by_msgs(self) -> dict:
        with open(self._file_path, 'r') as f:
            file_lines = f.readlines()
        output = {'0': []}
        current_msg = '0'
        frame_line_regex = r'^\s*(\d+)[ \t]+(-?(?:\d+(?:\.\d+)?|\.)?)[ \t]+(-?(?:\d+(?:\.\d+)?|\.)?)[ \t]+(-?(?:\d+(?:\.\d+)?|\.)?)[ \t]+(-?(?:\d+(?:\.\d+)?|\.)?)[ \t]+(-?(?:\d+(?:\.\d+)?|\.)?)[ \t]+(-?(?:\d+(?:\.\d+)?|\.)?)[ \t]+(-?(?:\d+(?:\.\d+)?|\.)?)[ \t]+([.A-Za-z]+)(?:[ \t]+[^\r\n]*)?$'
        while not file_lines[0].startswith('START'):
            file_lines = file_lines[1:]

        while not re.match(frame_line_regex, file_lines[0]):
            file_lines = file_lines[1:]

        for line in file_lines:
            if re.match(frame_line_regex, line):
                output[current_msg].append(line)
            elif re.match(r'MSG\t\d* \d', line):  # line.startswith('MSG'):
                current_msg = ''.join(line.split()[2:])
                output[current_msg] = []
            else:
                pass
                # print(f'ignored line: {line}')
        return output

    @staticmethod
    def __parse_asc_line(line: str) -> list:
        return line.replace('\t', ' ').split()[:-1]

    @property
    def csv_files(self):
        return self.__csv_files if self.__csv_files else self.split_asc_to_csvs_by_msgs()

    def split_asc_to_csvs_by_msgs(self, output_folder: str = '.') -> list:
        parts = self.__split_asc_by_msgs()
        os.makedirs(output_folder, exist_ok=True)
        output_csvs = []
        for trigger, lines in parts.items():
            csv_file = os.path.join(output_folder, self._file_path.split("\\")[-1].split(".")[0] + f'_{trigger}.csv')
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(range(8))
                writer.writerows([self.__parse_asc_line(line) for line in lines])
            output_csvs.append(CsvEyelinkSample(csv_file))
        self.__csv_files = output_csvs
        msg1 = self.__fix_no_msg_1()
        if msg1:
            output_csvs.insert(1, msg1)
        self.__csv_files = output_csvs
        return output_csvs

    def __fix_no_msg_1(self, msg_len: int = 20000):
        if list(filter(lambda f: os.path.basename(f.file_path).endswith('1.csv'), self.csv_files)):
            return None
        csv_to_split = list(filter(lambda f: os.path.basename(f.file_path).endswith('0.csv'), self.csv_files))[0]
        parsed = csv_to_split.parsed
        end_time = parsed[-1].time
        start_time = end_time - msg_len
        new_rows = list(filter(lambda row: row.time >= start_time, parsed))
        csv_file = csv_to_split.file_path[::-1].replace('0', '1', 1)[::-1]
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(range(8))
            writer.writerows([str(row).split('\t') for row in new_rows])
        return CsvEyelinkSample(csv_file)

    def analyze_trial(self, output_folder: str):
        self.split_asc_to_csvs_by_msgs(output_folder)
        with open(os.path.join(output_folder, f'{os.path.basename(self.file_path).split(".")[0]}_summary.csv'), 'w',
                  newline='') as summary_file:
            writer = csv.DictWriter(summary_file, fieldnames=['msg'] + list(self.csv_files[0].get_statistics().keys()))
            writer.writeheader()
            for csv_file in self.csv_files:
                stats_dict = csv_file.get_statistics()
                stats_dict['msg'] = csv_file.file_path.split('\\')[-1].split('.')[0][-1]
                writer.writerow(stats_dict)


class CsvEyelinkSample(EyelinkSample):
    def __init__(self, file_path: str):
        assert 'csv' == file_path.split('.')[-1], 'file type should be csv'
        super().__init__(file_path)
        self.__parsed = None

    @property
    def parsed(self) -> list:
        if self.__parsed:
            return self.__parsed

        self.__parsed = self.__parse_csv()
        return self.__parsed

    def __parse_csv(self) -> list:
        with open(self.file_path, 'r', newline='') as f:
            csv_reader = csv.reader(f)
            # skip header line
            next(csv_reader)
            return [Frame(row) for row in csv_reader]

    def get_blinks(self) -> list:
        blinks = []
        frames = self.parsed
        in_blink = False

        for frame in frames:
            if frame.is_blink:
                if not in_blink:
                    # starting a new blink
                    in_blink = True
                    start_time = end_time = frame.time
                else:
                    # continuing blink
                    end_time = frame.time
            else:
                if in_blink:
                    # ending blink
                    blinks.append(Fixation(start_time, end_time))
                    in_blink = False

        # close final blink if file ends mid-blink
        if in_blink:
            blinks.append(Fixation(start_time, end_time))

        return blinks

    def get_fixations_in_area(self, x: float, y: float, r: float) -> list:
        fixations = []
        was_last_frame_in_area = False
        for frame in self.parsed:
            # current frame isn't in the area
            if not frame.is_in_circle(x, y, r):
                if was_last_frame_in_area:
                    fixations.append(Fixation(start_time, end_time))
                was_last_frame_in_area = False
                continue

            # current frame is in the area, but previous wasn't
            if not was_last_frame_in_area:
                was_last_frame_in_area = True
                start_time, end_time = frame.time, frame.time
                continue

            # current frame in the area and so was the last one
            end_time = frame.time

        if was_last_frame_in_area:
            fixations.append(Fixation(start_time, end_time))

        return fixations

    def get_statistics(self) -> dict:
        statistics = {}
        blinks = self.get_blinks()
        center_fixations = self.get_fixations_in_area(SCREEN_CENTER_X, SCREEN_CENTER_Y, SCREEN_CENTER_R)
        # statistics['blinks'] = blinks
        statistics['blinks_count'] = len(blinks)
        statistics['blinks_total_duration'] = sum(list(map(lambda fixation: fixation.duration, blinks)))
        if len(blinks) != 0:
            statistics['blinks_avg_duration'] = statistics['blinks_total_duration'] / len(blinks)
            statistics['blinks_median_duration'] = sorted(list(map(lambda fixation: fixation.duration, blinks)))[
                len(blinks) // 2]
        else:
            statistics['blinks_avg_duration'] = -1
            statistics['blinks_median_duration'] = -1
        # statistics['center_fixations'] = center_fixations
        statistics['center_fixations_count'] = len(center_fixations)
        statistics['center_fixations_duration'] = sum(list(map(lambda fixation: fixation.duration, center_fixations)))
        if len(center_fixations) != 0:
            statistics['center_fixations_avg_duration'] = statistics['center_fixations_duration'] / len(
                center_fixations)
            statistics['center_fixations_median_duration'] = \
                sorted(list(map(lambda fixation: fixation.duration, center_fixations)))[len(center_fixations) // 2]
        else:
            statistics['center_fixations_avg_duration'] = -1
            statistics['center_fixations_median_duration'] = -1
        statistics['total_duration'] = self.parsed[-1].time - self.parsed[0].time
        # no eye tracking data was found
        if statistics['total_duration'] == statistics['blinks_total_duration']:
            statistics['center_fixations_percentage'] = 0
        else:
            statistics['center_fixations_percentage'] = statistics['center_fixations_duration'] / (statistics[
                'total_duration'] - statistics['blinks_total_duration']) * 100
        statistics['blinks_percentage'] = statistics['blinks_total_duration'] / statistics[
            'total_duration'] * 100
        return statistics
