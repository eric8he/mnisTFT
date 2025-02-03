#include <Arduino.h>
#include <pgmspace.h>
#include <math.h>     // for expf(), etc.

// Include the generated header from Python
#include "weights.h"

// ----------------------------------------------------------
// Model Architecture Recap (SmallNet):
//   conv1: input 1 -> 8 filters (3x3, stride=1)
//   relu
//   conv2: input 8 -> 16 filters (3x3, stride=1)
//   relu
//   max_pool2d(2x2)
//   flatten
//   fc1: (16*12*12) -> 64
//   relu
//   fc2: 64 -> 10
//   softmax
// ----------------------------------------------------------

// -------------------------
// Model Hyperparameters
// -------------------------
#define IMAGE_WIDTH    28
#define IMAGE_HEIGHT   28

// conv1: 1 -> 8, kernel=3, stride=1 => output shape: 8×26×26
#define CONV1_IN_CHANNELS     1
#define CONV1_OUT_CHANNELS    8
#define CONV1_KERNEL_SIZE     3
#define CONV1_STRIDE          1
#define CONV1_OUT_WIDTH       26
#define CONV1_OUT_HEIGHT      26

// conv2: 8 -> 16, kernel=3, stride=1 => output shape: 16×24×24
#define CONV2_IN_CHANNELS     8
#define CONV2_OUT_CHANNELS    16
#define CONV2_KERNEL_SIZE     3
#define CONV2_STRIDE          1
#define CONV2_OUT_WIDTH       24
#define CONV2_OUT_HEIGHT      24

// max_pool2d(2×2) => output shape: 16×12×12
#define POOL_SIZE       2
#define POOL_OUT_W      12
#define POOL_OUT_H      12

// fc1: (16*12*12) -> 64
#define FC1_IN_FEATURES  (16 * 12 * 12)
#define FC1_OUT_FEATURES 64

// fc2: 64 -> 10
#define FC2_IN_FEATURES  64
#define FC2_OUT_FEATURES 10

// ----------------------------------------------------------
// Function to read a quantized weight from PROGMEM
// and dequantize it to float
// ----------------------------------------------------------
static inline float readQuantWeight(const uint16_t* array, int index) {
    // Read the 16-bit stored weight from PROGMEM
    uint16_t rawU16 = pgm_read_word_near(&array[index]);
    // Reinterpret as signed int16
    int16_t rawI16 = (int16_t) rawU16;
    // Convert to float by dividing by the global scale
    float val = ((float) rawI16) / WEIGHT_SCALE;
    return val;
}

// ----------------------------------------------------------
// Activation function
// ----------------------------------------------------------
static inline float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

// ----------------------------------------------------------
// Naive conv2d (single batch, no padding)
// ----------------------------------------------------------
void conv2d(
  const float* inData,         
  float* outData,              
  int inC, int inH, int inW,
  int outC, int outH, int outW,
  int kernel_size, int stride,
  const uint16_t* weightArray, 
  const uint16_t* biasArray    
) {
    for (int oc = 0; oc < outC; oc++) {
        // Dequantize bias
        float b = readQuantWeight(biasArray, oc);

        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                float sum = 0.0f;
                for (int ic = 0; ic < inC; ic++) {
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            float val = inData[ic * inH * inW + ih * inW + iw];

                            // Weight index
                            int widx = oc*(inC*kernel_size*kernel_size)
                                       + ic*(kernel_size*kernel_size)
                                       + kh*(kernel_size)
                                       + kw;
                            
                            float w = readQuantWeight(weightArray, widx);
                            sum += val * w;
                        }
                    }
                }
                outData[oc*outH*outW + oh*outW + ow] = sum + b;
            }
        }
    }
}

// ----------------------------------------------------------
// Max-pool 2D
// ----------------------------------------------------------
void maxPool2d(
  const float* inData,
  float* outData,
  int channels,
  int inH, int inW,
  int poolSize,
  int outH, int outW
) {
    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                float maxVal = -999999.0f;
                for (int ph = 0; ph < poolSize; ph++) {
                    for (int pw = 0; pw < poolSize; pw++) {
                        int ih = oh * poolSize + ph;
                        int iw = ow * poolSize + pw;
                        float val = inData[c*inH*inW + ih*inW + iw];
                        if (val > maxVal) {
                            maxVal = val;
                        }
                    }
                }
                outData[c*outH*outW + oh*outW + ow] = maxVal;
            }
        }
    }
}

// ----------------------------------------------------------
// Linear layer
// ----------------------------------------------------------
void linear(
  const float* inData,
  float* outData,
  int inFeatures,
  int outFeatures,
  const uint16_t* weightArray,
  const uint16_t* biasArray
) {
    for (int of = 0; of < outFeatures; of++) {
        float sum = 0.0f;
        float b = readQuantWeight(biasArray, of);
        for (int inf = 0; inf < inFeatures; inf++) {
            float w = readQuantWeight(weightArray, of*inFeatures + inf);
            sum += inData[inf] * w;
        }
        outData[of] = sum + b;
    }
}

// ----------------------------------------------------------
// forwardPass: single 28×28 input -> 10-class output
// ----------------------------------------------------------
void forwardPass(const float* input, float* output)
{
    // 1) conv1 => 8×26×26
    static float conv1_out[CONV1_OUT_CHANNELS * CONV1_OUT_HEIGHT * CONV1_OUT_WIDTH];
    conv2d(
      input, conv1_out,
      CONV1_IN_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH,
      CONV1_OUT_CHANNELS, CONV1_OUT_HEIGHT, CONV1_OUT_WIDTH,
      CONV1_KERNEL_SIZE, CONV1_STRIDE,
      conv1_weight, conv1_bias
    );
    // relu
    for(int i = 0; i < (CONV1_OUT_CHANNELS * CONV1_OUT_HEIGHT * CONV1_OUT_WIDTH); i++){
        conv1_out[i] = relu(conv1_out[i]);
    }

    // 2) conv2 => 16×24×24
    static float conv2_out[CONV2_OUT_CHANNELS * CONV2_OUT_HEIGHT * CONV2_OUT_WIDTH];
    conv2d(
      conv1_out, conv2_out,
      CONV2_IN_CHANNELS, CONV1_OUT_HEIGHT, CONV1_OUT_WIDTH,
      CONV2_OUT_CHANNELS, CONV2_OUT_HEIGHT, CONV2_OUT_WIDTH,
      CONV2_KERNEL_SIZE, CONV2_STRIDE,
      conv2_weight, conv2_bias
    );
    // relu
    for(int i = 0; i < (CONV2_OUT_CHANNELS * CONV2_OUT_HEIGHT * CONV2_OUT_WIDTH); i++){
        conv2_out[i] = relu(conv2_out[i]);
    }

    // 3) max_pool2d => 16×12×12
    static float pool_out[CONV2_OUT_CHANNELS * POOL_OUT_H * POOL_OUT_W];
    maxPool2d(
      conv2_out, pool_out,
      CONV2_OUT_CHANNELS,
      CONV2_OUT_HEIGHT, CONV2_OUT_WIDTH,
      POOL_SIZE,
      POOL_OUT_H, POOL_OUT_W
    );

    // 4) flatten => size: 16*12*12 = 2304
    static float flat[FC1_IN_FEATURES];
    for(int i = 0; i < FC1_IN_FEATURES; i++){
        flat[i] = pool_out[i];
    }

    // 5) fc1 => 64
    static float fc1_out[FC1_OUT_FEATURES];
    linear(
      flat, fc1_out,
      FC1_IN_FEATURES, FC1_OUT_FEATURES,
      fc1_weight, fc1_bias
    );
    // relu
    for(int i = 0; i < FC1_OUT_FEATURES; i++){
        fc1_out[i] = relu(fc1_out[i]);
    }

    // 6) fc2 => 10
    static float fc2_out[FC2_OUT_FEATURES];
    linear(
      fc1_out, fc2_out,
      FC1_OUT_FEATURES, FC2_OUT_FEATURES,
      fc2_weight, fc2_bias
    );

    // 7) softmax
    float maxVal = fc2_out[0];
    for(int i = 1; i < FC2_OUT_FEATURES; i++){
        if(fc2_out[i] > maxVal) {
            maxVal = fc2_out[i];
        }
    }
    float sumExp = 0.0f;
    for(int i = 0; i < FC2_OUT_FEATURES; i++){
        sumExp += expf(fc2_out[i] - maxVal);
    }
    for(int i = 0; i < FC2_OUT_FEATURES; i++){
        output[i] = expf(fc2_out[i] - maxVal) / sumExp;
    }
}
