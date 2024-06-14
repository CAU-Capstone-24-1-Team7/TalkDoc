"""
Data preparation.

Author
------
Dongwon Kim, Dongwoo Kim 2021
N Park, WS Kang 2022 # for KdialectSpeech dataset, split wav file

apt install ffmpeg
pip install pydub # pydub는 ffmpeg를 필요로 함.
"""
import logging
import os
import glob
from pathlib import Path
import re

from speechbrain.dataio.dataio import load_pkl, merge_csvs, save_pkl
# from speechbrain.utils.data_utils import get_all_files

## Kang
import pandas as pd
import json
import os

# conda install -c conda-forge pydub
from pydub import AudioSegment
# conda install ffmpeg

logger = logging.getLogger(__name__)

OPT_FILE = "opt_kdialectspeech_prepare.pkl"
SAMPLERATE = 16000


def prepare_kdialectspeech(
    data_folder,
    splited_wav_folder,
    save_folder,
    province_code,
    data_ratio,
    select_n_sentences=None,
    skip_prep=False,
):
    """
    데이터 폴더, 지역코드 받아서,
    매니페스트 파일
    텍스트 파일
    만들기.
    

    확인 : 필요한 데이터 (지역별)폴더가 있는 지 확인


    This class prepares the csv files for the KdialectSpeech dataset.
    Download link: https://aihub.or.kr/aidata

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original KdialectSpeech dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    select_n_sentences : int
        Default : None
        If not None, only pick this many sentences.
    merge_lst : list
        List of KDialectSpeech splits (e.g, train, valid, test) to
        merge in a single csv file.
    merge_name: str
        Name of the merged csv file.
    skip_prep: bool
        If True, data preparation is skipped.


    Example
    -------
    >>> data_folder = 'datasets/KdialectSpeech'
    >>> tr_splits = ['train']
    >>> dev_splits = ['valid']
    >>> te_splits = ['test']
    >>> save_folder = 'KdialectSpeech_prepared'
    >>> prepare_kdialectspeech(data_folder, save_folder, tr_splits, dev_splits, \
                            te_splits)
    """

    if skip_prep:
        return
    data_folder = data_folder
    save_folder = save_folder
    select_n_sentences = select_n_sentences
    conf = {
        "select_n_sentences": select_n_sentences,
    }

    # Other variables
    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    ## province_folder = os.path.join(save_folder, province_code)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        # Path(save_folder).mkdir(parents=True, exist_ok=True)

    save_opt = os.path.join(save_folder, OPT_FILE)

    # Check if this phase is already done (if so, skip it)
    if skip(save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")

    # Additional checks to make sure the data folder contains kdialectspeech
    logger.info(f'data_folder : {data_folder}')
    check_kdialectspeech_folders(data_folder, province_code)

    create_csv(data_folder, save_folder, province_code, data_ratio, splited_wav_folder)

    # saving options
    save_pkl(conf, save_opt)


def skip(save_folder, conf):
    """
    province_code.csv가 있는지 확인

    Detect when the ksponspeech data prep can be skipped.

    Arguments
    ---------
    splits : list
        A list of the splits expected in the preparation.
    save_folder : str
        The location of the seave directory
    conf : dict
        The configuration options to ensure they haven't changed.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    logger.info(f'save_folder : {save_folder}')
    # Checking csv files
    skip = True

    if not os.path.isfile(os.path.join(save_folder, "train.csv")):
        skip = False

    if not os.path.isfile(os.path.join(save_folder, "valid.csv")):
        skip = False

    if not os.path.isfile(os.path.join(save_folder, "test.csv")):
        skip = False

    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    logger.info(f'skip : {skip}')
    return skip


def normalize(string):
    """
    This function normalizes a given string according to
    the normalization rule
    The normalization rule removes "/" indicating filler words,
    removes "+" indicating repeated words,
    removes all punctuation marks,
    removes non-speech symbols,
    and extracts orthographic transcriptions.

    Arguments
    ---------
    string : str
        The string to be normalized

    Returns
    -------
    str
        The string normalized according to the rules

    """
    # extracts orthographic transcription
    string = re.sub(r"\(([^)]*)\)\/\(([^)]*)\)", r"\1", string)
    # removes non-speech symbols
    string = re.sub(r"n/|b/|o/|l/|u/", "", string)
    # removes punctuation marks
    string = re.sub(r"[+*/.?!~(),]", "", string)
    # removes extra spaces
    string = re.sub(r"\s+", " ", string)
    string = string.strip()

    return string

### modified for KdialectSpeech
def check_kdialectspeech_folders(base_dir, province_code):
    """
    Check if the data folder actually contains the kdialectspeech dataset.

    If it does not, an error is raised.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If kdialectspeech is not found at the specified path.
    """
    # Checking if all the data directories exist
    if province_code == 'total':
        talk_dir = os.path.join(base_dir, 'people/talk/')
        say_dir = os.path.join(base_dir, 'person/say/')
        st_dir = os.path.join(base_dir, 'person/st/')
    else:
        talk_dir = os.path.join(base_dir, 'people/talk/' + province_code)
        say_dir = os.path.join(base_dir, 'person/say/' + province_code)
        st_dir = os.path.join(base_dir, 'person/st/' + province_code)
    
    if not os.path.exists(talk_dir):
        err_msg = (
            "the directory %s does not exist (it is expected in the "
            "kdialectspeech dataset)" % talk_dir
        )
        raise OSError(err_msg)
    
    if not os.path.exists(say_dir):
        err_msg = (
            "the directory %s does not exist (it is expected in the "
            "kdialectspeech dataset)" % talk_dir
        )
        raise OSError(err_msg)
    
    if not os.path.exists(st_dir):
        err_msg = (
            "the directory %s does not exist (it is expected in the "
            "kdialectspeech dataset)" % talk_dir
        )
        raise OSError(err_msg)

### for KdialectSpeech
def time_convert(str_time):
    """
    This gives time in milisecond

    Arguments
    ---------
    str_time : str
        The time in string type
        ex) 10:09:20.123

    str_time_format : str
    The time format in string type
    ex) "%H:%M:%S.%f"

    Returns
    -------
    float
        A milisecond time

    """    
    hh, mm, sec_mili = str_time.split(':')
    total_milis = float(hh) * 60 * 60 + float(mm) * 60 + float(sec_mili)
                
    return total_milis

def split_wav_file(input_audio, output_wav_file, start_time, end_time):
    """
    This gives time in milisecond

    Arguments
    ---------
    input_audio : audio signal
        audio signam from wav file

    output_wav_file : str
        wav file to save splited audio

    start_time : float
        start time to split in second

    end_time : float
        end time to split in second

    Returns
    -------
    float
        A milisecond time

    """
    start_time_milis = start_time * 1000
    end_time_milis = end_time * 1000
    splited_wav = input_audio[start_time_milis:end_time_milis]
    splited_wav.export(output_wav_file, format="wav")


def get_kdialectspeech_files(base_dir, province_code, file_ext='json'):
    """
    This gives directory names for kdialectspeech

    Arguments
    ---------
    base_dir : str
        The base directory of kdialectspeech data
    province_code : str
        The province code that one of ('gw', 'gs', 'jl', 'jj', 'cc')
    file_ext : str
        The file extention to be selected
        ex) 'json', 'wav'

    Returns
    -------
    list
        A list containing directories of the given data directory

    """
    if province_code == 'total':
        files = (
            glob.glob(os.path.join(base_dir, 'people/talk/*/*/*.' + file_ext))
            + glob.glob(os.path.join(base_dir, 'person/say/*/*/*.' + file_ext))
            + glob.glob(os.path.join(base_dir, 'person/st/*/*/*.' + file_ext))
        )
    else:
        files = (
            glob.glob(os.path.join(base_dir, 'people/talk/' + province_code + '/*/*.' + file_ext))
            + glob.glob(os.path.join(base_dir, 'person/say/' + province_code + '/*/*.' + file_ext))
            + glob.glob(os.path.join(base_dir, 'person/st/' + province_code + '/*/*.' + file_ext))
        )

    return files

def make_csv_lines(json_file_path, province_code, splited_wav_folder):
    """
    This gives manifest record to write in csv file.
    If the length of audio sound is longer than 30 sec, it split the audio file.

    Arguments
    ---------
    annotation_file_path : file path
        The annotation file name(input file) in json type
        ex) person/st/gs/collectorgs1/st_set1_collectorgs1_speakergs3_79_8.json

    splited_wav_folder : str
        The output folder to save splited file
        ex) "splited_file_dir"

    Returns
    -------
    None

    """

    csv_lines = []

    with open(json_file_path, encoding="UTF-8" ) as json_file :
        json_data = json.load(json_file)
        # data_file_name = json_data["fileName"]
        data_file_name = Path(json_file_path).stem # 확장자 제거
        dir = os.path.dirname(json_file_path)

        if splited_wav_folder == 'same':
            splited_wav_folder = dir

        os.makedirs(splited_wav_folder, exist_ok=True)

        input_wav_file = os.path.join(dir, data_file_name + '.wav')
        try:
            input_audio = AudioSegment.from_wav(input_wav_file)
        except:
            logger.info(f'wrong audio file : {input_wav_file}')
            return csv_lines

        sentences = json_data["transcription"]["sentences"]
        # print(f'number of sentences : {len(sentences)}')
        # print(data_file_name)

        for idx, sentence in enumerate(sentences) :
            #추가##############################
            #data_ duration
            data_start_time = sentence["startTime"]
            data_end_time = sentence["endTime"]
            
            start_time = time_convert(data_start_time)
            end_time = time_convert(data_end_time)
            duration = end_time - start_time
    
            # province_code = sentence["speakerId"][7:9]
            
            #wrd(dialect)
            dialect = sentence["dialect"]
            wrd = normalize(dialect)
            # wrd = dialect.translate(str.maketrans('', '', string.punctuation))\

            # if duration > 30:
            if duration > 25:
                logger.info(f'too long sentence included : {input_wav_file}')
                logger.info(f'sentence : {wrd} \'s duration : {duration}')
                continue
            
            if len(sentences) == 1:
                splited_file_name = data_file_name + '.wav'
                splited_file_path = os.path.join(dir, splited_file_name)
            else:
                splited_file_name = data_file_name + "-" + str(idx) + '.wav'
                splited_file_path = os.path.join(splited_wav_folder, splited_file_name)
                split_wav_file(input_audio, splited_file_path, start_time, end_time)

            duration = format(duration, '5.3f')            
            id = province_code + '-' + Path(splited_file_name).stem
            line = [id, duration, splited_file_path, province_code, wrd]
            csv_lines.append(line)
    return csv_lines


def create_csv(data_folder, save_folder, province_code, data_ratio, splited_wav_folder):
    """
    This makes manifest file from given base directory.

    Arguments
    ---------
    base_dir : base directory of KdialectSpeech
        ex) '/data/KdialectSpeech'

    province_code : str
        The province code that is one of ('gw', 'gs', 'jl', 'jj', 'cc')

    output_file : str
        The output file to save manifest
        ex) "output/manifest.csv"

    Returns
    -------
    None
    """

    json_file_list = get_kdialectspeech_files(data_folder, province_code, file_ext='json')

    total_csv_lines = []
    for json_file in json_file_list:
        total_csv_lines = total_csv_lines + make_csv_lines(json_file, province_code, splited_wav_folder) # [duration, splited_file_path,  province_code, wrd]

    total_csv_df = pd.DataFrame(total_csv_lines, columns=['ID', 'duration', 'wav', 'province_code', 'wrd'])
    total_csv_file = os.path.join(save_folder, 'total.csv')
    total_csv_df.to_csv(total_csv_file, encoding="UTF-8", index=False)

    # split total csv into train, valid, test
    line_number = len(total_csv_df)
    # data_ratio = {'tr':0.8, 'va':0.1, 'te':0.1} # yaml file에 설정
    train_number = round(line_number * data_ratio['tr'])
    valid_number = round(line_number * data_ratio['va'])
    # test_number = round(line_number * data_ratio['te'])

    train_data_df = total_csv_df[:train_number]
    train_csv_file = os.path.join(save_folder, 'train.csv')
    train_data_df.to_csv(train_csv_file, encoding="UTF-8", index=False)

    valid_data_df = total_csv_df[train_number:train_number + valid_number]
    valid_csv_file = os.path.join(save_folder, 'valid.csv')
    valid_data_df.to_csv(valid_csv_file, encoding="UTF-8", index=False)

    test_data_df = total_csv_df[train_number + valid_number:]
    test_csv_file = os.path.join(save_folder, 'test.csv')
    test_data_df.to_csv(test_csv_file, encoding="UTF-8", index=False)    


### txt file은 speechbrain.tokenizers.SentencePiece 에서 만듬.
    