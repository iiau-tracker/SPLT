from SPLT_tracker_new import MobileTracker
import argparse
import os
import cv2
import multiprocessing
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from toolkit.datasets import DatasetFactory
from toolkit.datasets.vot import VOTLTVideo
from common_path import *
parser = argparse.ArgumentParser(description='tracking')
parser.add_argument('--dataset', default= dataset_name_, type=str,
        help='eval one special dataset')
parser.add_argument('--video', default= video_name_, type=str,
        help='eval one special video')
parser.add_argument('--vis', default=False, help='whether visualzie result')
args = parser.parse_args()
def track_seq(tracker:MobileTracker, model_name:str, video:VOTLTVideo):
    print('processing ', video.name)
    toc = 0
    pred_bboxes = []
    scores = []
    track_times = []
    for idx, (img, gt_bbox) in enumerate(video):
        tic = cv2.getTickCount()
        if idx == 0:
            tracker.init_first(img, gt_bbox)
            pred_bbox = gt_bbox
            scores.append(None)
            if 'VOT2018-LT' == args.dataset:
                pred_bboxes.append([1])
            else:
                pred_bboxes.append(pred_bbox)
        else:
            # print('processing %d' % idx)
            outputs = tracker.track(img)
            pred_bbox = outputs[0]
            pred_bboxes.append(pred_bbox)
            scores.append(outputs[1])
        toc += cv2.getTickCount() - tic
        track_times.append((cv2.getTickCount() - tic) / cv2.getTickFrequency())
        if idx == 0:
            cv2.destroyAllWindows()
        if args.vis and idx > 0:
            gt_bbox = list(map(int, gt_bbox))
            pred_bbox = list(map(int, pred_bbox))
            cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                          (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 3)
            cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                          (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 3)
            cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow(video.name, img)
            cv2.waitKey(1)
    toc /= cv2.getTickFrequency()
    # save results
    if 'VOT2018-LT' == args.dataset:
        video_path = os.path.join('results', args.dataset, model_name,
                                  'longterm', video.name)
        if not os.path.isdir(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path,
                                   '{}_001.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x]) + '\n')
        result_path = os.path.join(video_path,
                                   '{}_001_confidence.value'.format(video.name))
        with open(result_path, 'w') as f:
            for x in scores:
                f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
        result_path = os.path.join(video_path,
                                   '{}_time.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in track_times:
                f.write("{:.6f}\n".format(x))


def main():
    use_vot = False
    display = False
    threads = 8
    # load config
    model_name = 'SPLT36_haojie_mutil'
    dataset_root = '/media/masterbin-iiau/WIN_SSD/VOT2018_LT35'
    # create model
    tracker = MobileTracker(vot=use_vot, dis=display)
    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)
    param_list = [(tracker,model_name,video) for _, video in enumerate(dataset)]
    # OPE tracking
    '''我看到Martin的pytracking里使用了多进程,我也想试试看,但是遇到了如下错误'''
    with multiprocessing.Pool(processes=threads) as pool:
        ''' Following TypeError is encountered
        TypeError: can't pickle _thread.RLock objects'''
        pool.starmap(track_seq, param_list)




if __name__ == '__main__':
    main()
