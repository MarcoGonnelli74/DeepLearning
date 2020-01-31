/**
 * Copyright (c) 2017-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#include <cstring>
#include <iostream>
#include "nvdsinfer_custom_impl.h"

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLIP(a,min,max) (MAX(MIN(a, max), min))

/* This is a sample bounding box parsing function for the sample Resnet10
 * detector model provided with the SDK. */

/* C-linkage to prevent name-mangling */
extern "C"
bool NvDsInferParseCustomDetectnet (std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferParseObjectInfo> &objectList)
{
  static NvDsInferDimsCHW covLayerDims;
  static NvDsInferDimsCHW bboxLayerDims;
  static int bboxLayerIndex = -1;
  static int covLayerIndex = -1;
  static bool classMismatchWarn = false;
  int numClassesToParse;

  /* Find the bbox layer */

  if (bboxLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "bboxes") == 0) {
        bboxLayerIndex = i;
        getDimsCHWFromDims(bboxLayerDims, outputLayersInfo[i].dims);
        break;
      }
    }
    if (bboxLayerIndex == -1) {
    std::cerr << "Could not find bbox layer buffer while parsing" << std::endl;
    return false;
    }
  }

  /* Find the cov layer */
  if (covLayerIndex == -1) {
    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
      if (strcmp(outputLayersInfo[i].layerName, "coverage") == 0) {
        covLayerIndex = i;
        getDimsCHWFromDims(covLayerDims, outputLayersInfo[i].dims);
        break;
      }
    }
    if (covLayerIndex == -1) {
    std::cerr << "Could not find cov layer buffer while parsing" << std::endl;
    return false;
    }
  }

  /* Warn in case of mismatch in number of classes */
  if (!classMismatchWarn) {
    if (covLayerDims.c != (int) detectionParams.numClassesConfigured) {
      std::cerr << "WARNING: Num classes mismatch. Configured:" <<
        detectionParams.numClassesConfigured << ", detected by network: " <<
        covLayerDims.c << std::endl;
    }
    classMismatchWarn = true;
  }

  std::cerr << "-----------------------------------------------" << std::endl;
  std::cerr << "--- COVLAYERIDX: " << covLayerIndex << " ---" << std::endl;
  std::cerr << "-----------------------------------------------" << std::endl;

  /* Calculate the number of classes to parse */
  numClassesToParse = MIN (covLayerDims.c,
      (int) detectionParams.numClassesConfigured);

  int gridW = covLayerDims.w;
  int gridH = covLayerDims.h;
  int gridSize = gridW * gridH;
  float *outputCovBuf = (float *) outputLayersInfo[covLayerIndex].buffer;
  float *outputBboxBuf = (float *) outputLayersInfo[bboxLayerIndex].buffer;
  int cell_width  = networkInfo.width  / gridW;
  int cell_height = networkInfo.height / gridH;


  std::cerr << "-----------------------------------------------" << std::endl;
  std::cerr << "--- NETWORK INFO WIDTH : " << networkInfo.width << " HEIGHT : " << networkInfo.height <<
  	 " NUM-CLASSES->" << numClassesToParse << std::endl;

  for (int c = 0; c < numClassesToParse; c++)
  {
    float *outputX1 = outputBboxBuf; // + (c * 1 * bboxLayerDims.h * bboxLayerDims.w);

    float *outputY1 = outputX1 + gridSize;
    float *outputX2 = outputY1 + gridSize;
    float *outputY2 = outputX2 + gridSize;

    float threshold = detectionParams.perClassThreshold[c];

    std::cerr << "-----------------------------------------------" << std::endl;
    std::cerr << "--- CLASS INFO ID : " << c << " THRESHOLD : " << threshold <<  std::endl;
    std::cerr << "--- GRIDW : " << gridW << " GRIDH : " << gridH <<  std::endl;
    std::cerr << "--- GRIDSIZE : " << gridSize <<  std::endl;
    std::cerr << "-----------------------------------------------" << std::endl;

    for (int h = 0; h < gridH; h++)
    {
      for (int w = 0; w < gridW; w++)
      {
        int i = w + h * gridW;

        //std::cerr << "--- outputCovBuf " << outputCovBuf[c * gridSize + i] << " ---" <<  std::endl;

        if (outputCovBuf[c * gridSize + i] >= threshold)
        {
          //std::cerr << "--- outputCovBuf magg TH " << outputCovBuf[c * gridSize + i] << " ---" <<  std::endl;

          NvDsInferParseObjectInfo object;
          float rectX1f, rectY1f, rectX2f, rectY2f;

          object.classId = c;
          object.detectionConfidence = outputCovBuf[c * gridSize + i];

          float mx = w * cell_width;
          float my = h * cell_height;

          rectX1f = outputX1[w + h * gridW] + mx;
          rectY1f = outputY1[w + h * gridW] + my;
          rectX2f = outputX2[w + h * gridW] + mx;
          rectY2f = outputY2[w + h * gridW] + my;

          //Clip object box co-ordinates to network resolution
          object.left = CLIP(rectX1f, 0, networkInfo.width - 1);
          object.top = CLIP(rectY1f, 0, networkInfo.height - 1);
          object.width = CLIP(rectX2f, 0, networkInfo.width - 1) -
                             object.left + 1;
          object.height = CLIP(rectY2f, 0, networkInfo.height - 1) -
                             object.top + 1;

	  std::cerr << "Bounding BOX DETECTED ClassID:"<< object.classId << std::endl;
	  std::cerr << " Confidence:" << object.detectionConfidence << std::endl <<
	  " BBOX COORDS: ["<< object.left << "," << object.top << "," << object.width << "," << object.height << "]"
          << std::endl;

          objectList.push_back(object);
        }
      }
    }
  }

  //std::cerr << "-----------------------------------------------" << std::endl;
  //std::cerr << "-----------------------------------------------" << std::endl;

  return true;
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomDetectnet);
