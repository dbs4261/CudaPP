//
// Created by Daniel Simon on 3/31/20.
//

#ifndef CUDAPP_STREAM_MANAGMENT_H
#define CUDAPP_STREAM_MANAGMENT_H

#include "cudapp/utilities/ide_helpers.h"

namespace cudapp {

// Make streams spawned by a device?

// cudaStreamAddCallback (maybe allow it? maybe put this in kernel launchers)
// cudaStreamAttachMemAsync
// cudaStreamBeginCapture (this goes in the graph section)
// cudaStreamCreate
// cudaStreamCreateWithFlags
// cudaStreamCreateWithPriority
// cudaStreamDestroy
// cudaStreamEndCapture (this goes in the graph section)
// cudaStreamGetFlags
// cudaStreamGetPriority
// cudaStreamIsCapturing (this goes in the graph section)
// cudaStreamQuery
// cudaStreamSynchronize
// cudaStreamWaitEvent
// cudaThreadExchangeStreamCaptureMode (this goes in the graph section)

}

#endif //CUDAPP_STREAM_MANAGMENT_H
