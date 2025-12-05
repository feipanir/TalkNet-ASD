import os, subprocess, glob, pandas, tqdm, cv2, numpy
from scipy.io import wavfile
import argparse, warnings
import pdb


def init_args(args):
    # The details for the following folders/files can be found in the annotation of the function 'preprocess_AVA' below
    args.modelSavePath    = os.path.join(args.savePath, 'model')
    args.scoreSavePath    = os.path.join(args.savePath, 'score.txt')
    args.trialPathAVA     = os.path.join(args.dataPathAVA, 'csv')
    args.audioOrigPathAVA = os.path.join(args.dataPathAVA, 'orig_audios')
    args.visualOrigPathAVA= os.path.join(args.dataPathAVA, 'orig_videos')
    args.audioPathAVA     = os.path.join(args.dataPathAVA, 'clips_audios')
    args.visualPathAVA    = os.path.join(args.dataPathAVA, 'clips_videos')
    args.trainTrialAVA    = os.path.join(args.trialPathAVA, 'train_loader.csv')

    if args.evalDataType == 'val':
        args.evalTrialAVA = os.path.join(args.trialPathAVA, 'val_loader.csv')
        args.evalOrig     = os.path.join(args.trialPathAVA, 'val_orig.csv')  
        args.evalCsvSave  = os.path.join(args.savePath,     'val_res.csv') 
    else:
        args.evalTrialAVA = os.path.join(args.trialPathAVA, 'test_loader.csv')
        args.evalOrig     = os.path.join(args.trialPathAVA, 'test_orig.csv')    
        args.evalCsvSave  = os.path.join(args.savePath,     'test_res.csv')
    
    os.makedirs(args.modelSavePath, exist_ok = True)
    os.makedirs(args.dataPathAVA, exist_ok = True)
    return args
 

def download_pretrain_model_AVA():
    if os.path.isfile('pretrain_AVA.model') == False:
        Link = "1NVIkksrD3zbxbDuDbPc_846bLfPSZcZm"
        cmd = "gdown --id %s -O %s"%(Link, 'pretrain_AVA.model')
        subprocess.call(cmd, shell=True, stdout=None)

def preprocess_AVA(args):
    # This preprocesstion is modified based on this [repository](https://github.com/fuankarion/active-speakers-context).
    # The required space is 302 G. 
    # If you do not have enough space, you can delate `orig_videos`(167G) when you get `clips_videos(85G)`.
    #                             also you can delate `orig_audios`(44G) when you get `clips_audios`(6.4G).
    # So the final space is less than 100G.
    # The AVA dataset will be saved in 'AVApath' folder like the following format:
    # ```
    # ├── clips_audios  (The audio clips cut from the original movies)
    # │   ├── test
    # │   ├── train
    # │   └── val
    # ├── clips_videos (The face clips cut from the original movies, be save in the image format, frame-by-frame)
    # │   ├── test
    # │   ├── train
    # │   └── val
    # ├── csv
    # │   ├── test_file_list.txt (name of the test videos)
    # │   ├── test_loader.csv (The csv file we generated to load data for testing)
    # │   ├── test_orig.csv (The combination of the given test csv files)
    # │   ├── train_loader.csv (The csv file we generated to load data for training)
    # │   ├── train_orig.csv (The combination of the given training csv files)
    # │   ├── trainval_file_list.txt (name of the train/val videos)
    # │   ├── val_loader.csv (The csv file we generated to load data for validation)
    # │   └── val_orig.csv (The combination of the given validation csv files)
    # ├── orig_audios (The original audios from the movies)
    # │   ├── test
    # │   └── trainval
    # └── orig_videos (The original movies)
    #     ├── test
    #     └── trainval
    # ```

    # download_csv(args) # Take 1 minute 
    # download_videos(args) # Take 6 hours
    # import pdb; pdb.set_trace()
    # extract_audio(args) # Take 1 hour
    # extract_audio_clips(args) # Take 3 minutes
    # extract_audio_chunking_clips(args) # Take 3 minutes
    # import pdb; pdb.set_trace()
    extract_video_clips(args) # Take about 2 days

def download_csv(args): 
    # Take 1 minute to download the required csv files
    '''
    Link = "1C1cGxPHaJAl1NQ2i7IhRgWmdvsPhBCUy"
    cmd = "gdown --id %s -O %s"%(Link, args.dataPathAVA + '/csv.tar.gz')
    subprocess.call(cmd, shell=True, stdout=None)
    cmd = "tar -xzvf %s -C %s"%(args.dataPathAVA + '/csv.tar.gz', args.dataPathAVA)
    subprocess.call(cmd, shell=True, stdout=None)
    os.remove(args.dataPathAVA + '/csv.tar.gz')
    '''
    pass

def download_videos(args): 
    # Take 6 hours to download the original movies, follow this repository: https://github.com/cvdfoundation/ava-dataset
    for dataType in ['trainval', 'test']:
        fileList = open('%s/%s_file_list.txt'%(args.trialPathAVA, dataType)).read().splitlines()
        outFolder = '%s/%s'%(args.visualOrigPathAVA, dataType)
        for fileName in fileList:
            dstPath = os.path.join(outFolder, fileName)
            if os.path.isfile(dstPath):
                continue
            cmd = "wget -P %s https://s3.amazonaws.com/ava-dataset/%s/%s"%(outFolder, dataType, fileName)
            subprocess.call(cmd, shell=True, stdout=None)

def extract_audio(args):
    # Take 1 hour to extract the audio from movies
    for dataType in ['trainval', 'test']:
        inpFolder = '%s/%s'%(args.visualOrigPathAVA, dataType)
        outFolder = '%s/%s'%(args.audioOrigPathAVA, dataType)
        os.makedirs(outFolder, exist_ok = True)
        videos = glob.glob("%s/*"%(inpFolder))
        for videoPath in tqdm.tqdm(videos):
            audioPath = '%s/%s'%(outFolder, videoPath.split('/')[-1].split('.')[0] + '.wav')
            cmd = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads 8 %s -loglevel panic" % (videoPath, audioPath))
            subprocess.call(cmd, shell=True, stdout=None)


def extract_audio_clips(args):
    # Take 3 minutes to extract the audio clips

    dic = {'train':'trainval', 'val':'trainval', 'test':'test'}
    for dataType in [ 'train', 'val', 'test']:
        # import pdb; pdb.set_trace()
        print("dataType:", (dataType))
        df = pandas.read_csv(os.path.join(args.trialPathAVA, '%s_orig.csv'%(dataType)), engine='python')
        dfNeg = pandas.concat([df[df['label_id'] == 0], df[df['label_id'] == 2]])
        dfPos = df[df['label_id'] == 1]
        insNeg = dfNeg['instance_id'].unique().tolist()
        insPos = dfPos['instance_id'].unique().tolist()
        df = pandas.concat([dfPos, dfNeg]).reset_index(drop=True)
        df = df.sort_values(['entity_id', 'frame_timestamp']).reset_index(drop=True)
        entityList = df['entity_id'].unique().tolist()
        df = df.groupby('entity_id')
        audioFeatures = {}
        outDir = os.path.join(args.audioPathAVA, dataType)
        audioDir = os.path.join(args.audioOrigPathAVA, dic[dataType])
        for l in df['video_id'].unique().tolist():
            d = os.path.join(outDir, l[0])
            if not os.path.isdir(d):
                os.makedirs(d)
        for entity in tqdm.tqdm(entityList, total = len(entityList)):
            insData = df.get_group(entity)
            videoKey = insData.iloc[0]['video_id']
            start = insData.iloc[0]['frame_timestamp']
            end = insData.iloc[-1]['frame_timestamp']
            entityID = insData.iloc[0]['entity_id']
            insPath = os.path.join(outDir, videoKey, entityID+'.wav')
            if videoKey not in audioFeatures.keys():                
                audioFile = os.path.join(audioDir, videoKey+'.wav')
                sr, audio = wavfile.read(audioFile)
                audioFeatures[videoKey] = audio
            audioStart = int(float(start)*sr)
            audioEnd = int(float(end)*sr)
            audioData = audioFeatures[videoKey][audioStart:audioEnd]
            wavfile.write(insPath, sr, audioData)

import gc   # <<< IMPORTANT
def extract_audio_chunking_clips(args):

    dic = {'train': 'trainval', 'val': 'trainval', 'test': 'test'}

    # Process train / val / test
    for dataType in ['train', 'val', 'test']:
        csv_path = os.path.join(args.trialPathAVA, f"{dataType}_orig.csv")
        audioDir = os.path.join(args.audioOrigPathAVA, dic[dataType])
        outDir = os.path.join(args.audioPathAVA, dataType)

        # Cache audio files (still per-video)
        audioFeatures = {}

        print(f"\n>>> Processing {csv_path} in chunks...")

        # Read CSV chunk-by-chunk (low memory)
        chunk_iter = pandas.read_csv(
            csv_path,
            engine="c",
            low_memory=True,
            chunksize=100,   # safe chunk size
            usecols=[
                "label_id",
                "instance_id",
                "entity_id",
                "frame_timestamp",
                "video_id"
            ],
        )
        print(chunk_iter.engine)

        chunk_idx = 0

        # Process each chunk independently
        for chunk in chunk_iter:
            # import pdb; pdb.set_trace()
            chunk_idx += 1
            print(f"--- Chunk {chunk_idx} --- Rows: {len(chunk)}")

            # Split pos/neg like original code
            dfNeg = pandas.concat([chunk[chunk['label_id'] == 0],
                               chunk[chunk['label_id'] == 2]])
            dfPos = chunk[chunk['label_id'] == 1]

            # Keep instance lists if needed downstream (optional)
            insNeg = dfNeg['instance_id'].unique().tolist()
            insPos = dfPos['instance_id'].unique().tolist()

            # Recombine pos/neg
            df = pandas.concat([dfPos, dfNeg]).reset_index(drop=True)

            # Sort & group inside chunk
            df = df.sort_values(['entity_id', 'frame_timestamp']).reset_index(drop=True)

            # Entities inside this chunk
            entityList = df['entity_id'].unique().tolist()

            df_grouped = df.groupby('entity_id')

            # Pre-create folders
            # import pdb; pdb.set_trace()
            for videoKey in df['video_id'].unique().tolist():
                d = os.path.join(outDir, videoKey)
                if videoKey in ['Ekwy7wzLfjc']:
                    continue
                if not os.path.isdir(d):
                    os.makedirs(d, exist_ok=True)

            # Process each entity in the chunk
            for entity in tqdm.tqdm(entityList, total=len(entityList)):
                insData = df_grouped.get_group(entity)

                videoKey = insData.iloc[0]['video_id']
                if videoKey in ['Ekwy7wzLfjc']:
                    continue
                start = insData.iloc[0]['frame_timestamp']
                end   = insData.iloc[-1]['frame_timestamp']
                entityID = insData.iloc[0]['entity_id']

                outPath = os.path.join(outDir, videoKey, f"{entityID}.wav")

                # Load audio only once per video
                if videoKey not in audioFeatures:
                    audioFile = os.path.join(audioDir, f"{videoKey}.wav")
                    sr, audio = wavfile.read(audioFile)
                    audioFeatures[videoKey] = audio

                audio = audioFeatures[videoKey]

                # Slice audio
                audioStart = int(float(start) * sr)
                audioEnd   = int(float(end)   * sr)
                audioData  = audio[audioStart:audioEnd]

                # Save clip
                wavfile.write(outPath, sr, audioData)

            # ======== CLEAN UP VARIABLES FOR THIS CHUNK ========
            del dfNeg
            del dfPos
            del insNeg
            del insPos
            del df
            del df_grouped
            del entityList
            del chunk

            gc.collect()   # <<< Forces memory release! 

        print(f"Finished processing {dataType}.")

def extract_video_clips(args):
    # Take about 2 days to crop the face clips.
    # You can optimize this code to save time, while this process is one-time.
    # If you do not need the data for the test set, you can only deal with the train and val part. That will take 1 day.
    # This procession may have many warning info, you can just ignore it.
    dic = {'train':'trainval', 'val':'trainval', 'test':'test'}
    for dataType in ['train', 'val', 'test']:
        df = pandas.read_csv(os.path.join(args.trialPathAVA, '%s_orig.csv'%(dataType)))
        dfNeg = pandas.concat([df[df['label_id'] == 0], df[df['label_id'] == 2]])
        dfPos = df[df['label_id'] == 1]
        insNeg = dfNeg['instance_id'].unique().tolist()
        insPos = dfPos['instance_id'].unique().tolist()
        df = pandas.concat([dfPos, dfNeg]).reset_index(drop=True)
        df = df.sort_values(['entity_id', 'frame_timestamp']).reset_index(drop=True)
        entityList = df['entity_id'].unique().tolist()
        df = df.groupby('entity_id')
        outDir = os.path.join(args.visualPathAVA, dataType)
        audioDir = os.path.join(args.visualOrigPathAVA, dic[dataType])
        # import pdb; pdb.set_trace()
        for l in df['video_id'].unique().tolist():
            d = os.path.join(outDir, l[0])
            if not os.path.isdir(d):
                os.makedirs(d)
        for entity in tqdm.tqdm(entityList, total = len(entityList)):
            insData = df.get_group(entity)
            videoKey = insData.iloc[0]['video_id']
            entityID = insData.iloc[0]['entity_id']
            videoDir = os.path.join(args.visualOrigPathAVA, dic[dataType])
            videoFile = glob.glob(os.path.join(videoDir, '{}.*'.format(videoKey)))[0]
            V = cv2.VideoCapture(videoFile)
            insDir = os.path.join(os.path.join(outDir, videoKey, entityID))
            # import pdb; pdb.set_trace()
            if not os.path.isdir(insDir):
                os.makedirs(insDir)
            j = 0
            for _, row in insData.iterrows():
                imageFilename = os.path.join(insDir, str("%.2f"%row['frame_timestamp'])+'.jpg')
                # Check if file exists
                if os.path.exists(imageFilename):
                    print("imgFile exist:", imageFilename)
                    continue  # Skip this iteration and go to the next row

                # import pdb; pdb.set_trace()
                V.set(cv2.CAP_PROP_POS_MSEC, row['frame_timestamp'] * 1e3)
                _, frame = V.read()
                h = numpy.size(frame, 0)
                w = numpy.size(frame, 1)
                x1 = int(row['entity_box_x1'] * w)
                y1 = int(row['entity_box_y1'] * h)
                x2 = int(row['entity_box_x2'] * w)
                y2 = int(row['entity_box_y2'] * h)
                face = frame[y1:y2, x1:x2, :]
                j = j+1
                cv2.imwrite(imageFilename, face)


def main():
    # pip install pandas
    # pip install opencv-python
    # pip install scipy
    # sudo apt install ffmpeg

    # The structure of this code is learnt from https://github.com/clovaai/voxceleb_trainer
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description = "TalkNet Training")
    # Training details
    parser.add_argument('--lr',           type=float, default=0.0001,help='Learning rate')
    parser.add_argument('--lrDecay',      type=float, default=0.95,  help='Learning rate decay rate')
    parser.add_argument('--maxEpoch',     type=int,   default=25,    help='Maximum number of epochs')
    parser.add_argument('--testInterval', type=int,   default=1,     help='Test and save every [testInterval] epochs')
    parser.add_argument('--batchSize',    type=int,   default=2500,  help='Dynamic batch size, default is 2500 frames, other batchsize (such as 1500) will not affect the performance')
    parser.add_argument('--nDataLoaderThread', type=int, default=4,  help='Number of loader threads')
    # Data path
    parser.add_argument('--dataPathAVA',  type=str, default="../AVADataPath", help='Save path of AVA dataset')
    parser.add_argument('--savePath',     type=str, default="exps/exp1")
    # Data selection
    parser.add_argument('--evalDataType', type=str, default="val", help='Only for AVA, to choose the dataset for evaluation, val or test')
    # For download dataset only, for evaluation only
    parser.add_argument('--downloadAVA',     dest='downloadAVA', action='store_true', help='Only download AVA dataset and do related preprocess')
    parser.add_argument('--evaluation',      dest='evaluation', action='store_true', help='Only do evaluation by using pretrained model [pretrain_AVA.model]')
    args = parser.parse_args()
    # Data loader
    args = init_args(args)
    # import pdb; pdb.set_trace()
    preprocess_AVA(args)


if __name__ == "__main__":
    main()
