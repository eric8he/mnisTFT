/*******************************************************************************
 *  Example: 32x32 Black/White Canvas with Touch Drawing and Calibration
 *******************************************************************************/

#include "FS.h"        // For SPIFFS
#include <SPI.h>
#include <TFT_eSPI.h>  // Bodmer's TFT library

/************  TFT and Touch Setup  ************/

TFT_eSPI tft = TFT_eSPI();

// File name used to store the calibration data in SPIFFS
#define CALIBRATION_FILE "/TouchCalData1"
// Repeat calibration if set to true
#define REPEAT_CAL false

#define WIDTH 320
#define HEIGHT 480

//----------------------------------------------------//
//              32x32 CANVAS PARAMETERS               //
//----------------------------------------------------//
#define CANVAS_SIZE    32
#define BLOCK_SIZE     8   // Each logical pixel is a 8×8 block
#define CANVAS_MARGIN  (WIDTH - CANVAS_SIZE * BLOCK_SIZE) / 2  
bool canvas[CANVAS_SIZE][CANVAS_SIZE];

//----------------------------------------------------//
//            BUTTON PARAMETERS (OPTIONAL)            //
//   (from your example; can be removed if desired)    //
//----------------------------------------------------//
#define KEY_X 160       // Centre of key
#define KEY_Y 50
#define KEY_W 320       // Width
#define KEY_H 22        // Height
#define KEY_SPACING_X 0
#define KEY_SPACING_Y 1
#define KEY_TEXTSIZE 1
#define NUM_KEYS 6

TFT_eSPI_Button key[NUM_KEYS];

//----------------------------------------------------//
//                   SETUP FUNCTION                   //
//----------------------------------------------------//
void setup() {
  Serial.begin(115200);

  tft.init();
  tft.setRotation(2);  // Adjust rotation to suit your screen orientation

  // If you have a backlight pin defined in the TFT_eSPI setup:
  #ifndef TFT_BL
    Serial.println("No TFT backlight pin defined");
  #else
    pinMode(TFT_BL, OUTPUT);
    digitalWrite(TFT_BL, HIGH);
  #endif

  // Perform touchscreen calibration
  touch_calibrate();

  // Initialize the screen and fill it black
  tft.fillScreen(TFT_BLACK);

  // Initialize our 32×32 canvas to all-white (false means white)
  for (int y = 0; y < CANVAS_SIZE; y++) {
    for (int x = 0; x < CANVAS_SIZE; x++) {
      canvas[x][y] = false;
    }
  }


  // Draw a frame around the canvas
  tft.drawRect(CANVAS_MARGIN - 1, CANVAS_MARGIN - 1, CANVAS_SIZE * BLOCK_SIZE + 2, CANVAS_SIZE * BLOCK_SIZE + 2, TFT_WHITE);

  // Draw the (all-white) canvas
  drawCanvas();


  // (Optional) Draw some demo buttons
  tft.setFreeFont(&FreeMono9pt7b);
  //drawButtons();
}

//----------------------------------------------------//
//                    MAIN LOOP                       //
//----------------------------------------------------//
void loop() {
  // 1. Handle any touchscreen presses
  uint16_t t_x = 0, t_y = 0;
  bool pressed = tft.getTouch(&t_x, &t_y);

  if (pressed) {
    // Map touch coordinates to our 32×32 “logical” grid
    int gridX = (t_x - CANVAS_MARGIN) / BLOCK_SIZE;  // each block is 6 px wide
    int gridY = (t_y - CANVAS_MARGIN) / BLOCK_SIZE;  // each block is 6 px tall

    // Only draw if within bounds
    if (gridX >= 0 && gridX < CANVAS_SIZE &&
        gridY >= 0 && gridY < CANVAS_SIZE) {
      // Example: paint black when touched
      setPixel(gridX, gridY, true);
    }

    delay(7);
  }
}

/*******************************************************************************
 *                      CANVAS DRAWING FUNCTIONS
 ******************************************************************************/

/**
 * Draws the entire 32×32 canvas. Each logical pixel is a 6×6 block.
 */
void drawCanvas() {
  for (int y = 0; y < CANVAS_SIZE; y++) {
    for (int x = 0; x < CANVAS_SIZE; x++) {
      drawBlock(x, y, canvas[x][y]);
    }
  }
}

/**
 * Draws one logical pixel (6×6 block) at (x, y).
 * colorValue = true (black) or false (white).
 */
void drawBlock(int x, int y, bool colorValue) {
  uint16_t color = colorValue ? TFT_WHITE : TFT_BLACK;
  int16_t screenX = x * BLOCK_SIZE + CANVAS_MARGIN;
  int16_t screenY = y * BLOCK_SIZE + CANVAS_MARGIN;
  tft.fillRect(screenX, screenY, BLOCK_SIZE, BLOCK_SIZE, color);
}

/**
 * Sets the logical pixel in the canvas array and updates only that block.
 */
void setPixel(int x, int y, bool color) {
  if (x < 0 || x >= CANVAS_SIZE || y < 0 || y >= CANVAS_SIZE) return;
  // Set the canvas cell
  canvas[x][y] = color;
  // Redraw just that cell
  drawBlock(x, y, color);
}

/*******************************************************************************
 *                         OPTIONAL BUTTONS
 *   (Derived from your provided example, can be removed if not needed)
 ******************************************************************************/
void drawButtons() {
  for (int i = 0; i < NUM_KEYS; i++) {
    key[i].initButton(
      &tft,
      KEY_X,                                // x center
      KEY_Y + i*(KEY_H + KEY_SPACING_Y),    // y center
      KEY_W,                                // width
      KEY_H,                                // height
      TFT_BLACK,                            // outline
      TFT_CYAN,                             // fill
      TFT_BLACK,                            // text
      "",                                   // label
      KEY_TEXTSIZE
    );
    // Position label text relative to ML_DATUM (middle-left)
    key[i].setLabelDatum(i * 10 - (KEY_W / 2), 0, ML_DATUM);
    // Draw button with dynamic label
    key[i].drawButton(false, "ML_DATUM + " + (String)(i * 10) + "px");
  }
}

/*******************************************************************************
 *                 TOUCHSCREEN CALIBRATION FUNCTION
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
    // Recalibrate if we want to repeat
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

  // Calibration data valid
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
    // Finally apply calibration data
    tft.setTouch(calData);
  }
}