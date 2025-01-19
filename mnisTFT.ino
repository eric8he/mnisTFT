// /*******************************************************************************
//  *  Example: 32x32 Black/White Canvas with Touch Drawing and Calibration
//  *******************************************************************************/

// #include "FS.h"        // For SPIFFS
// #include <SPI.h>
// #include <TFT_eSPI.h>  // Bodmer's TFT library
// #include "nn.h"
// #include "weights.h"

// /************  TFT and Touch Setup  ************/

// TFT_eSPI tft = TFT_eSPI();

// // File name used to store the calibration data in SPIFFS
// #define CALIBRATION_FILE "/TouchCalData1"
// // Repeat calibration if set to true
// #define REPEAT_CAL false

// #define WIDTH 320
// #define HEIGHT 480

// //----------------------------------------------------//
// //                 CANVAS PARAMETERS                  //
// //----------------------------------------------------//
// #define CANVAS_SIZE    28
// #define BLOCK_SIZE     10   // Each logical pixel is a 10×10 block
// #define CANVAS_MARGIN  (WIDTH - CANVAS_SIZE * BLOCK_SIZE) / 2  
// bool canvas[CANVAS_SIZE][CANVAS_SIZE];

// //----------------------------------------------------//
// //            BUTTON PARAMETERS (OPTIONAL)            //
// //   (from your example; can be removed if desired)    //
// //----------------------------------------------------//
// #define KEY_X 160       // Centre of key
// #define KEY_Y 50
// #define KEY_W 320       // Width
// #define KEY_H 22        // Height
// #define KEY_SPACING_X 0
// #define KEY_SPACING_Y 1
// #define KEY_TEXTSIZE 1
// #define NUM_KEYS 6

// TFT_eSPI_Button key[NUM_KEYS];

// // ----------------------------------------------------------
// // Example input image (28×28). For a real application,
// // you would fill this with sensor/camera data, etc.
// // ----------------------------------------------------------


// //----------------------------------------------------//
// //                   SETUP FUNCTION                   //
// //----------------------------------------------------//
// void setup() {
//   Serial.begin(115200);
  
//   while(!Serial) {
//     delay(10);
//   }

//   float output[10];
//   forwardPass(example_correct, output);

//   Serial.println("Classification Results:");
//   for(int i=0; i<10; i++){
//     Serial.print("Class ");
//     Serial.print(i);
//     Serial.print(": ");
//     Serial.println(output[i], 6);
//   }

//   tft.init();
//   tft.setRotation(2);  // Adjust rotation to suit your screen orientation

//   // If you have a backlight pin defined in the TFT_eSPI setup:
//   #ifndef TFT_BL
//     Serial.println("No TFT backlight pin defined");
//   #else
//     pinMode(TFT_BL, OUTPUT);
//     digitalWrite(TFT_BL, HIGH);
//   #endif

//   // Perform touchscreen calibration
//   touch_calibrate();

//   // Initialize the screen and fill it black
//   tft.fillScreen(TFT_BLACK);

//   // Initialize our 32×32 canvas to all-white (false means white)
//   for (int y = 0; y < CANVAS_SIZE; y++) {
//     for (int x = 0; x < CANVAS_SIZE; x++) {
//       canvas[x][y] = false;
//     }
//   }


//   // Draw a frame around the canvas
//   tft.drawRect(CANVAS_MARGIN - 1, CANVAS_MARGIN - 1, CANVAS_SIZE * BLOCK_SIZE + 2, CANVAS_SIZE * BLOCK_SIZE + 2, TFT_WHITE);

//   // Draw the (all-white) canvas
//   drawCanvas();


//   // (Optional) Draw some demo buttons
//   tft.setFreeFont(&FreeMono9pt7b);
//   //drawButtons();
// }

// //----------------------------------------------------//
// //                    MAIN LOOP                       //
// //----------------------------------------------------//
// void loop() {
//   // 1. Handle any touchscreen presses
//   uint16_t t_x = 0, t_y = 0;
//   bool pressed = tft.getTouch(&t_x, &t_y);

//   if (pressed) {
//     // Map touch coordinates to our 32×32 “logical” grid
//     int gridX = (t_x - CANVAS_MARGIN) / BLOCK_SIZE;  // each block is 6 px wide
//     int gridY = (t_y - CANVAS_MARGIN) / BLOCK_SIZE;  // each block is 6 px tall

//     // Only draw if within bounds
//     if (gridX >= 0 && gridX < CANVAS_SIZE &&
//         gridY >= 0 && gridY < CANVAS_SIZE) {
//       // Example: paint black when touched
//       delay(10);
//       setPixel(gridX, gridY, true);
//     }
//   }
// }

// /*******************************************************************************
//  *                      CANVAS DRAWING FUNCTIONS
//  ******************************************************************************/

// /**
//  * Draws the entire 32×32 canvas. Each logical pixel is a 6×6 block.
//  */
// void drawCanvas() {
//   for (int y = 0; y < CANVAS_SIZE; y++) {
//     for (int x = 0; x < CANVAS_SIZE; x++) {
//       drawBlock(x, y, canvas[x][y]);
//     }
//   }
// }

// /**
//  * Draws one logical pixel (6×6 block) at (x, y).
//  * colorValue = true (black) or false (white).
//  */
// void drawBlock(int x, int y, bool colorValue) {
//   uint16_t color = colorValue ? TFT_WHITE : TFT_BLACK;
//   int16_t screenX = x * BLOCK_SIZE + CANVAS_MARGIN;
//   int16_t screenY = y * BLOCK_SIZE + CANVAS_MARGIN;
//   tft.fillRect(screenX, screenY, BLOCK_SIZE, BLOCK_SIZE, color);
// }

// /**
//  * Sets the logical pixel in the canvas array and updates only that block.
//  */
// void setPixel(int x, int y, bool color) {
//   if (x < 0 || x >= CANVAS_SIZE || y < 0 || y >= CANVAS_SIZE) return;
//   // Set the canvas cell
//   canvas[x][y] = color;
//   // Redraw just that cell
//   drawBlock(x, y, color);
// }

// /*******************************************************************************
//  *                         OPTIONAL BUTTONS
//  *   (Derived from your provided example, can be removed if not needed)
//  ******************************************************************************/
// void drawButtons() {
//   for (int i = 0; i < NUM_KEYS; i++) {
//     key[i].initButton(
//       &tft,
//       KEY_X,                                // x center
//       KEY_Y + i*(KEY_H + KEY_SPACING_Y),    // y center
//       KEY_W,                                // width
//       KEY_H,                                // height
//       TFT_BLACK,                            // outline
//       TFT_CYAN,                             // fill
//       TFT_BLACK,                            // text
//       "",                                   // label
//       KEY_TEXTSIZE
//     );
//     // Position label text relative to ML_DATUM (middle-left)
//     key[i].setLabelDatum(i * 10 - (KEY_W / 2), 0, ML_DATUM);
//     // Draw button with dynamic label
//     key[i].drawButton(false, "ML_DATUM + " + (String)(i * 10) + "px");
//   }
// }

// /*******************************************************************************
//  *                 TOUCHSCREEN CALIBRATION FUNCTION
//  ******************************************************************************/
// void touch_calibrate() {
//   uint16_t calData[5];
//   uint8_t calDataOK = 0;

//   // Check that SPIFFS is working
//   if (!SPIFFS.begin()) {
//     Serial.println("Formatting SPIFFS...");
//     SPIFFS.format();
//     SPIFFS.begin();
//   }

//   // Check if calibration file exists and size is correct
//   if (SPIFFS.exists(CALIBRATION_FILE)) {
//     // Recalibrate if we want to repeat
//     if (REPEAT_CAL) {
//       SPIFFS.remove(CALIBRATION_FILE);
//     }
//     else {
//       File f = SPIFFS.open(CALIBRATION_FILE, "r");
//       if (f) {
//         if (f.readBytes((char*)calData, 14) == 14) {
//           calDataOK = 1;
//         }
//         f.close();
//       }
//     }
//   }

//   // Calibration data valid
//   if (calDataOK && !REPEAT_CAL) {
//     tft.setTouch(calData);
//   } else {
//     // Do calibration
//     tft.fillScreen(TFT_BLACK);
//     tft.setCursor(20, 0);
//     tft.setTextFont(2);
//     tft.setTextSize(1);
//     tft.setTextColor(TFT_WHITE, TFT_BLACK);

//     tft.println("Touch the corners as indicated");

//     tft.setTextFont(1);
//     tft.println();

//     if (REPEAT_CAL) {
//       tft.setTextColor(TFT_RED, TFT_BLACK);
//       tft.println("Set REPEAT_CAL to false to stop this from running again!");
//     }

//     // Calibrate
//     tft.calibrateTouch(calData, TFT_MAGENTA, TFT_BLACK, 15);

//     tft.setTextColor(TFT_GREEN, TFT_BLACK);
//     tft.println("Calibration complete!");

//     // Save calibration data
//     File f = SPIFFS.open(CALIBRATION_FILE, "w");
//     if (f) {
//       f.write((const unsigned char*)calData, 14);
//       f.close();
//     }
//     // Finally apply calibration data
//     tft.setTouch(calData);
//   }
// }

/*******************************************************************************
 *  Example: 28×28 Black/White Canvas with Touch Drawing and Classification
 *******************************************************************************/

#include "FS.h"        // For SPIFFS
#include <SPI.h>
#include <TFT_eSPI.h>  // Bodmer's TFT library

#include "nn.h"        // forwardPass(...) declaration
#include "weights.h"   // conv1_weight, fc1_bias, etc.

//--------------------------------------------
// TFT and Touch Setup
//--------------------------------------------
TFT_eSPI tft = TFT_eSPI();

// File name used to store the calibration data in SPIFFS
#define CALIBRATION_FILE "/TouchCalData1"
#define REPEAT_CAL false

#define WIDTH 320
#define HEIGHT 480

//--------------------------------------------
// CANVAS PARAMETERS
//--------------------------------------------
#define CANVAS_SIZE   28
#define BLOCK_SIZE    10   // Each "pixel" is a 10×10 block on screen
#define CANVAS_MARGIN ((WIDTH - (CANVAS_SIZE * BLOCK_SIZE)) / 2)

// 2D array storing the user’s drawing (black/white)
bool canvas[CANVAS_SIZE][CANVAS_SIZE];

//--------------------------------------------
// CLASSIFY BUTTON PARAMETERS
//--------------------------------------------
#define BTN_X      80
#define BTN_Y      400
#define BTN_W      160
#define BTN_H      40
#define BTN_TEXT   "CLASSIFY"

bool wasPressed = false; // Track last touch state

//--------------------------------------------
// Forward Declarations
//--------------------------------------------
void touch_calibrate();
void drawCanvas();
void drawBlock(int x, int y, bool colorValue);
void setPixel(int x, int y, bool color);
void classifyCanvas();  // Our new function

//--------------------------------------------
// SETUP
//--------------------------------------------
void setup() {
  Serial.begin(115200);
  while(!Serial) {
    delay(10);
  }

  tft.init();
  tft.setRotation(2); // Adjust as needed for your display orientation

#ifdef TFT_BL
  pinMode(TFT_BL, OUTPUT);
  digitalWrite(TFT_BL, HIGH);
#endif

  // Touch calibration
  touch_calibrate();

  // Clear screen
  tft.fillScreen(TFT_BLACK);

  // Initialize the canvas to all white (false = white, true = black)
  for (int y = 0; y < CANVAS_SIZE; y++) {
    for (int x = 0; x < CANVAS_SIZE; x++) {
      canvas[x][y] = false; // White
    }
  }

  // Draw a frame around the canvas
  tft.drawRect(
    CANVAS_MARGIN - 1, 
    CANVAS_MARGIN - 1, 
    (CANVAS_SIZE * BLOCK_SIZE) + 2, 
    (CANVAS_SIZE * BLOCK_SIZE) + 2, 
    TFT_WHITE
  );

  // Draw the initial (all-white) canvas
  drawCanvas();

  // Draw the "Classify" button
  tft.drawRect(BTN_X, BTN_Y, BTN_W, BTN_H, TFT_WHITE);
  tft.setCursor(BTN_X + 10, BTN_Y + 10);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  tft.setTextSize(2);
  tft.print(BTN_TEXT);
}

//--------------------------------------------
// LOOP
//--------------------------------------------
void loop() {
  uint16_t t_x = 0, t_y = 0;
  bool pressed = tft.getTouch(&t_x, &t_y);

  if (pressed) {
    // If we just transitioned from not-pressed -> pressed
    if (!wasPressed) {
      wasPressed = true;  // Touch is now active

      // Check if user tapped the "Classify" button region
      if (t_x >= BTN_X && t_x < (BTN_X + BTN_W) &&
          t_y >= BTN_Y && t_y < (BTN_Y + BTN_H)) {
        // Classify the drawn image
        classifyCanvas();
      }
      else {
        // Otherwise, check if tapped within the 28×28 drawing canvas
        int gridX = (t_x - CANVAS_MARGIN) / BLOCK_SIZE;
        int gridY = (t_y - CANVAS_MARGIN) / BLOCK_SIZE;
        if (gridX >= 0 && gridX < CANVAS_SIZE &&
            gridY >= 0 && gridY < CANVAS_SIZE) {
          // Paint black
          setBrushPlus(gridX, gridY, true);
        }
      }
    }
    else {
      // (User might be dragging or continuing to press)
      // For real "drawing," you might keep painting black as user drags:
      int gridX = (t_x - CANVAS_MARGIN) / BLOCK_SIZE;
      int gridY = (t_y - CANVAS_MARGIN) / BLOCK_SIZE;
      if (gridX >= 0 && gridX < CANVAS_SIZE &&
          gridY >= 0 && gridY < CANVAS_SIZE) {
        setBrushPlus(gridX, gridY, true);
      }
    }
  }
  else {
    // No press detected
    wasPressed = false;
  }
}

/*******************************************************************************
 *  CANVAS DRAWING FUNCTIONS
 ******************************************************************************/

void drawCanvas() {
  for (int y = 0; y < CANVAS_SIZE; y++) {
    for (int x = 0; x < CANVAS_SIZE; x++) {
      drawBlock(x, y, canvas[x][y]);
    }
  }
}

// colorValue = true => black pixel, false => white pixel
void drawBlock(int x, int y, bool colorValue) {
  uint16_t color = colorValue ? TFT_WHITE : TFT_BLACK;
  int16_t screenX = (x * BLOCK_SIZE) + CANVAS_MARGIN;
  int16_t screenY = (y * BLOCK_SIZE) + CANVAS_MARGIN;
  tft.fillRect(screenX, screenY, BLOCK_SIZE, BLOCK_SIZE, color);
}

void setPixel(int x, int y, bool color) {
  if (x < 0 || x >= CANVAS_SIZE || y < 0 || y >= CANVAS_SIZE) return;
  canvas[x][y] = color;
  drawBlock(x, y, color);
}

void setBrushPlus(int cx, int cy, bool color) {
  // Center
  setPixel(cx, cy, color);

  // Up
  if (cy - 1 >= 0)
    setPixel(cx, cy - 1, color);

  // Down
  if (cy + 1 < CANVAS_SIZE)
    setPixel(cx, cy + 1, color);

  // Left
  if (cx - 1 >= 0)
    setPixel(cx - 1, cy, color);

  // Right
  if (cx + 1 < CANVAS_SIZE)
    setPixel(cx + 1, cy, color);
}

/*******************************************************************************
 *  CLASSIFY THE CURRENT CANVAS
 ******************************************************************************/
void classifyCanvas() {
  // 1. Convert the boolean canvas[][] to a float array (28×28)
  static float inputImage[CANVAS_SIZE * CANVAS_SIZE];
  for (int y = 0; y < CANVAS_SIZE; y++) {
    for (int x = 0; x < CANVAS_SIZE; x++) {
      // If black => 1.0f, if white => 0.0f (or vice versa, depending on training)
      inputImage[y * CANVAS_SIZE + x] = (canvas[x][y]) ? 1.0f : 0.0f;
    }
  }

  // 2. Run inference
  float output[10];
  forwardPass(inputImage, output);

  // 3. Find the most likely class
  float bestVal = output[0];
  int bestIdx = 0;
  for (int i = 1; i < 10; i++) {
    if (output[i] > bestVal) {
      bestVal = output[i];
      bestIdx = i;
    }
  }

  // 4. Print classification result and probabilities
  //    Clear a region below the button for text:
  tft.fillRect(0, BTN_Y + BTN_H + 5, WIDTH, 60, TFT_BLACK);
  tft.setCursor(10, BTN_Y + BTN_H + 10);
  tft.setTextSize(2);
  tft.setTextColor(TFT_GREEN, TFT_BLACK);
  tft.printf("Prediction: %d (Prob=%.3f)\r\n", bestIdx, bestVal);

  // (Optional) Print each probability
  // Move cursor down a bit
  tft.setCursor(10, BTN_Y + BTN_H + 35);
  tft.setTextSize(1);
  tft.setTextColor(TFT_WHITE, TFT_BLACK);
  for (int i = 0; i < 10; i++) {
    tft.printf(" %d: %.3f  ", i, output[i]);
  }
  Serial.println(); // also print to Serial
}

/*******************************************************************************
 *  TOUCHSCREEN CALIBRATION FUNCTION
 ******************************************************************************/
void touch_calibrate() {
  uint16_t calData[5];
  uint8_t calDataOK = 0;

  // Check that SPIFFS is working
  if (!SPIFFS.begin()) {
    Serial.println("Formatting SPIFFS...");
    SPIFFS.format();
    SPIFFS.begin();
  }

  // Check if calibration file exists and size is correct
  if (SPIFFS.exists(CALIBRATION_FILE)) {
    if (REPEAT_CAL) {
      SPIFFS.remove(CALIBRATION_FILE);
    }
    else {
      File f = SPIFFS.open(CALIBRATION_FILE, "r");
      if (f) {
        if (f.readBytes((char*)calData, 14) == 14) {
          calDataOK = 1;
        }
        f.close();
      }
    }
  }

  if (calDataOK && !REPEAT_CAL) {
    tft.setTouch(calData);
  } else {
    // Do calibration
    tft.fillScreen(TFT_BLACK);
    tft.setCursor(20, 0);
    tft.setTextFont(2);
    tft.setTextSize(1);
    tft.setTextColor(TFT_WHITE, TFT_BLACK);
    tft.println("Touch the corners as indicated");
    tft.setTextFont(1);
    tft.println();

    if (REPEAT_CAL) {
      tft.setTextColor(TFT_RED, TFT_BLACK);
      tft.println("Set REPEAT_CAL to false to stop this from running again!");
    }

    // Calibrate
    tft.calibrateTouch(calData, TFT_MAGENTA, TFT_BLACK, 15);

    tft.setTextColor(TFT_GREEN, TFT_BLACK);
    tft.println("Calibration complete!");

    // Save calibration data
    File f = SPIFFS.open(CALIBRATION_FILE, "w");
    if (f) {
      f.write((const unsigned char*)calData, 14);
      f.close();
    }
    tft.setTouch(calData);
  }
}
