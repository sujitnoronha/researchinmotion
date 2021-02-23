
import logging
import threading
import os
import sys
from collections import deque
from argparse import ArgumentParser, SUPPRESS
from math import exp as exp
from time import perf_counter
from enum import Enum
from tensorflow.keras import layers
import tensorflow as tf

import cv2
import numpy as np
from openvino.inference_engine import IECore
from utils_openvino import * 
import monitors


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input", help="Required. Path to an image/video file. (Specify 'cam' to work with "
                                            "camera)", required=True, type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with "
                           "the kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
                           " acceptable. The sample will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Labels mapping file", default=None, type=str)
    args.add_argument("-t", "--prob_threshold", help="Optional. Probability threshold for detections filtering",
                      default=0.5, type=float)
    args.add_argument("-iout", "--iou_threshold", help="Optional. Intersection over union threshold for overlapping "
                                                       "detections filtering", default=0.4, type=float)
    args.add_argument("-r", "--raw_output_message", help="Optional. Output inference results raw values showing",
                      default=False, action="store_true")
    args.add_argument("-nireq", "--num_infer_requests", help="Optional. Number of infer requests",
                      default=1, type=int)
    args.add_argument("-nstreams", "--num_streams",
                      help="Optional. Number of streams to use for inference on the CPU or/and GPU in throughput mode "
                           "(for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> "
                           "or just <nstreams>)",
                      default="", type=str)
    args.add_argument("-nthreads", "--number_threads",
                      help="Optional. Number of threads to use for inference on CPU (including HETERO cases)",
                      default=None, type=int)
    args.add_argument("-loop_input", "--loop_input", help="Optional. Iterate over input infinitely",
                      action='store_true')
    args.add_argument("-no_show", "--no_show", help="Optional. Don't show output", action='store_true')
    args.add_argument('-u', '--utilization_monitors', default='', type=str,
                      help='Optional. List of monitors to show initially.')
    args.add_argument("--keep_aspect_ratio", action="store_true", default=False,
                      help='Optional. Keeps aspect ratio on resize.')
    return parser


def Check(a,  b):
    dist = ((a[0] - b[0]) ** 2 + 550 / ((a[1] + b[1]) / 2) * (a[1] - b[1]) ** 2) ** 0.5
    calibration = (a[1] + b[1]) / 2
    if 0 < dist < 0.25 * calibration:
        return True
    else:
        return False




def main():

    args = build_argparser().parse_args()
    print("loading classifier.........")
    model = tf.keras.models.load_model('./model/Mobilenet_17_11_20.h5')
    model.summary()
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255.,
                                   )

    lb = ['Fire', 'Normal Car', 'Normal', 'Road Accident', 'Shooting', 'Violence']

    mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
    Q = deque(maxlen=128)


    print('Creating inference Engine')
    ie = IECore()

    config_user_specified = {}
    config_min_latency = {}
    devices_nstreams = {}
    if args.num_streams:
        devices_nstreams = {device: args.num_streams for device in ['CPU', 'GPU'] if device in args.device} \
                           if args.num_streams.isdigit() \
                           else dict([device.split(':') for device in args.num_streams.split(',')])

    if 'CPU' in args.device:
        if args.cpu_extension:
            ie.add_extension(args.cpu_extension,'CPU')
        if args.number_threads is not None:
            config_user_specified['CPU_THREAD_NUM'] = str(args.number_threads)

        if 'CPU' in devices_nstreams:
            config_user_specified['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] \
                                                              if int(devices_nstreams['CPU']) > 0 \
                                                              else 'CPU_THROUGHPUT_AUTO'

        config_min_latency['CPU_THROUGHPUT_STREAMS'] = '1'

        
    if 'GPU' in args.device:
        if 'GPU' in devices_nstreams:
            config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                                                              if int(devices_nstreams['GPU']) > 0 \
                                                              else 'GPU_THROUGHPUT_AUTO'

        config_min_latency['GPU_THROUGHPUT_STREAMS'] = '1'
 
    #------------------------------------------------------------------------------------------------------------

    
    print("loading yolo v4-tiny")

    net = ie.read_network(args.model, 'frozen_darknet_yolov4_model.bin')

    if "CPU" in args.device:
        supported_layers = ie.query_network(net,"CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    assert len(net.input_info) == 1, "Sample supports only YOLO V3 based single input topologies"



    #---------------------------------------------------------------------------------------------------------------

    print("inputs")
    print(net.input_info)
    input_blob = next(iter(net.input_info))


    #read and pre-process input images
    if net.input_info[input_blob].input_data.shape[1] == 3:
        input_height, input_width = net.input_info[input_blob].input_data.shape[2:]
        nchw_shape = True
    else:
        input_height, input_width = net.input_info[input_blob].input_data.shape[1:3]
        nchw_shape = False
    
    
    labels_map = None

    input_stream = 0 

    mode = Mode(Modes.USER_SPECIFIED)
    cap = cv2.VideoCapture('vid/vid_short.mp4')
    wait_key_time = 1

    print("loading model to plugin")
    exec_nets = {}

    exec_nets[Modes.USER_SPECIFIED] = ie.load_network(network = net, device_name= args.device,
                                    config= config_user_specified,
                                    num_requests=args.num_infer_requests)
    
    exec_nets[Modes.MIN_LATENCY] = ie.load_network(network=net, device_name=args.device.split(":")[-1].split(",")[0],
                                                config=config_min_latency,
                                                num_requests=1)

    empty_requests = deque(exec_nets[mode.current].requests)
    completed_request_results = {}
    next_frame_id = 0
    next_frame_id_to_show = 0
    mode_info = { mode.current: ModeInfo() }
    event = threading.Event()
    callback_exceptions = []

    # ----------------------------------------------- 6. Doing inference -----------------------------------------------
    print("Starting inference...")
    
    presenter = monitors.Presenter(args.utilization_monitors, 55,
        (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 4), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 8)))

    
    while (cap.isOpened() \
           or completed_request_results \
           or len(empty_requests) < len(exec_nets[mode.current].requests)) \
          and not callback_exceptions:
    
        if next_frame_id_to_show in completed_request_results:
            frame, output, start_time, is_same_mode = completed_request_results.pop(next_frame_id_to_show)
        
            next_frame_id_to_show += 1
            if is_same_mode: 
                mode_info[mode.current].frames_count += 1

            objects = get_objects(output, net, (input_height, input_width), frame.shape[:-1], args.prob_threshold,
                                  args.keep_aspect_ratio)
            objects = filter_objects(objects, args.iou_threshold, args.prob_threshold)

            
            origin_im_size = frame.shape[:-1]
            presenter.drawGraphs(frame)

            pairs = []
            center = []
            status = []
            boxes =[]
            for obj in objects:
                # Validation bbox of detected object
                xmax = min(obj['xmax'], origin_im_size[1])
                ymax = min(obj['ymax'], origin_im_size[0])
                xmin = max(obj['xmin'], 0)
                ymin = max(obj['ymin'], 0)
                color = (min(obj['class_id'] * 12.5, 255),
                         min(obj['class_id'] * 7, 255),
                         min(obj['class_id'] * 5, 255))
                det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else \
                    str(obj['class_id'])

                boxes.append([xmax, ymax, xmin, ymin])
                center.append([int(xmin + xmax /2), int(ymin + ymax/2)])
                status.append(False)
            
            for i in range(len(center)):
                for j in range(len(center)):
                    close = Check(center[i], center[j])

                    if close:
                        pairs.append([center[i], center[j]])
                        status[i] = True
                        status[j] = True    
            index = 0
            for i in range(len(boxes)):
                if status[index] == True:
                    cv2.rectangle(frame, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])), (0, 0, 150), 2)
                elif status[index] == False:
                    cv2.rectangle(frame,(int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])), (0, 255, 0), 2)
                
                index += 1
            for h in pairs:
                cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
        

            canvas = np.zeros((250, 300, 3), dtype="uint8")
            frame1 = frame.copy()
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame1 = cv2.resize(frame1, (224, 224)).astype("float32")
            frame1 = train_datagen.standardize(frame1)   

            
            preds = model.predict(np.expand_dims(frame1, axis=0),workers=8,use_multiprocessing=True)[0]
            Q.append(preds)
            

            for (i,(lab, prob)) in enumerate(zip(lb, preds)):
                text= "{}:{:.2f}%".format(lab, prob*100)
                w = int(prob*300)
                cv2.rectangle(canvas, (7, (i*35) +5), 
                    (w, (i*35)+35), (0,0,255), -1)
                cv2.putText(canvas, text, (10,(i*35)+23), cv2.FONT_HERSHEY_SIMPLEX,0.45, (255,255,255),2)

            results = np.array(Q).mean(axis=0)
            i = np.argmax(results)
            label = lb[i]
            print(label)
            # draw the activity on the output frame
            text = "{}".format(label)
            cv2.putText(frame, text, (105, 50), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 255, 0), 2)
            
            cv2.imshow("probs", canvas)




            if mode_info[mode.current].frames_count != 0:
                fps_message = "FPS: {:.1f}".format(mode_info[mode.current].frames_count / \
                                                   (perf_counter() - mode_info[mode.current].last_start_time))
            
                put_highlighted_text(frame, fps_message, (15, 20), cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)
            
            if not args.no_show:
                cv2.imshow("Detection Results",frame)
                key = cv2.waitKey(wait_key_time)

                if key in {ord("q"), ord("Q"), 27}: # ESC key
                    break
                if key == 9: # Tab key
                    prev_mode = mode.current
                    mode.next()

                    await_requests_completion(exec_nets[prev_mode].requests)
                    empty_requests.clear()
                    empty_requests.extend(exec_nets[mode.current].requests)

                    mode_info[prev_mode].last_end_time = perf_counter()
                    mode_info[mode.current] = ModeInfo()
                else:
                    presenter.handleKey(key)

        elif empty_requests and cap.isOpened():
            start_time = perf_counter()
            ret, frame = cap.read()
            if not ret:
                if args.loop_input:
                    cap.open(input_stream)
                else:
                    cap.release()
                continue

            request = empty_requests.popleft()

            # resize input_frame to network size
            in_frame = preprocess_frame(frame, input_height, input_width, nchw_shape, args.keep_aspect_ratio)

            # Start inference
            request.set_completion_callback(py_callback=async_callback,
                                            py_data=(request,
                                                     next_frame_id,
                                                     mode.current,
                                                     frame,
                                                     start_time,
                                                     completed_request_results,
                                                     empty_requests,
                                                     mode,
                                                     event,
                                                     callback_exceptions))
            request.async_infer(inputs={input_blob: in_frame})
            next_frame_id += 1

        else:
            event.wait()

    if callback_exceptions:
        raise callback_exceptions[0]
    
    for exec_net in exec_nets.values():
        await_requests_completion(exec_net.requests)


if __name__ == '__main__':
    sys.exit(main() or 0)
