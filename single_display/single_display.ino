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

bool wasPressed = false; // Track last touch state
bool buttonWasPressed = false;

//--------------------------------------------
// Forward Declarations
//--------------------------------------------
void touch_calibrate();
void drawCanvas();
void setPixel(int x, int y, bool color);
void classifyCanvas();

// Global mutex for TFT access
SemaphoreHandle_t tftMutex = NULL;

// Classification task handle
TaskHandle_t classifyTaskHandle = NULL;

// Struct for passing data between cores
struct ClassificationRequest {
  float image[CANVAS_SIZE * CANVAS_SIZE];
  volatile bool pending;
  volatile bool processing;
} classificationData;

// Mode button parameters
#define MODE_BTN_SIZE 30
#define MODE_BTN_MARGIN 5
#define MODE_BTN_X (CANVAS_MARGIN + (CANVAS_SIZE * BLOCK_SIZE) - MODE_BTN_SIZE - MODE_BTN_MARGIN)
#define MODE_BTN_Y CANVAS_MARGIN + MODE_BTN_MARGIN

bool drawMode = true;  // true = draw (white), false = erase (black)

void drawModeButton() {
  if (xSemaphoreTake(tftMutex, portMAX_DELAY)) {
    // Draw button background
    tft.fillRect(MODE_BTN_X, MODE_BTN_Y, MODE_BTN_SIZE, MODE_BTN_SIZE, drawMode ? TFT_WHITE : TFT_BLACK);
    
    // Draw button border
    tft.drawRect(MODE_BTN_X, MODE_BTN_Y, MODE_BTN_SIZE, MODE_BTN_SIZE, TFT_WHITE);
    
    // Draw letter
    tft.setTextColor(drawMode ? TFT_BLACK : TFT_WHITE);
    tft.setTextSize(2);
    tft.setCursor(MODE_BTN_X + 8, MODE_BTN_Y + 8);
    tft.print(drawMode ? "D" : "E");

    xSemaphoreGive(tftMutex);
  }
}

//--------------------------------------------
// SETUP
//--------------------------------------------
void setup() {
  Serial.begin(115200);
  while(!Serial) {
    delay(10);
  }

  Serial.println("serial started");

  // Create mutex for TFT access before initializing TFT
  tftMutex = xSemaphoreCreateMutex();
  
  Serial.println("mutex taken");
  tft.init();
  tft.setRotation(0);

#ifdef TFT_BL
  pinMode(TFT_BL, OUTPUT);
  digitalWrite(TFT_BL, HIGH);
#endif

  // Touch calibration
  touch_calibrate();

  // Clear screen to black
  tft.fillScreen(TFT_BLACK);

  // Draw a frame around the canvas
  tft.drawRect(
    CANVAS_MARGIN - 1, 
    CANVAS_MARGIN - 1, 
    (CANVAS_SIZE * BLOCK_SIZE) + 2, 
    (CANVAS_SIZE * BLOCK_SIZE) + 2, 
    TFT_WHITE
  );

  // Fill canvas area with black
  tft.fillRect(
    CANVAS_MARGIN,
    CANVAS_MARGIN,
    CANVAS_SIZE * BLOCK_SIZE,
    CANVAS_SIZE * BLOCK_SIZE,
    TFT_BLACK
  );

  // Initialize the canvas to all black (true = white, false = black)
  for (int y = 0; y < CANVAS_SIZE; y++) {
    for (int x = 0; x < CANVAS_SIZE; x++) {
      canvas[x][y] = false; // Black
    }
  }
  
  // Draw initial mode button
  drawModeButton();
  
  // Clear classification flags
  classificationData.pending = false;
  classificationData.processing = false;
  
  // Create classification task on core 1
  xTaskCreatePinnedToCore(
    classifyTask,        // Function to implement the task
    "classifyTask",      // Name of the task
    8192,               // Stack size in words
    NULL,               // Task input parameter
    1,                  // Priority of the task
    &classifyTaskHandle, // Task handle
    1                   // Core where the task should run
  );
}

//--------------------------------------------
// LOOP
//--------------------------------------------
void loop() {
  uint16_t t_x = 0, t_y = 0;
  bool pressed = false;
  
  if (xSemaphoreTake(tftMutex, portMAX_DELAY)) {
    pressed = tft.getTouch(&t_x, &t_y);
    xSemaphoreGive(tftMutex);
  }

  if (pressed) {
    // If we just transitioned from not-pressed -> pressed
    if (!wasPressed) {
      wasPressed = true;  // Touch is now active
      
      // Check if mode button was pressed
      if (t_x >= MODE_BTN_X && t_x < (MODE_BTN_X + MODE_BTN_SIZE) &&
          t_y >= MODE_BTN_Y && t_y < (MODE_BTN_Y + MODE_BTN_SIZE)) {
        buttonWasPressed = true;
        drawMode = !drawMode;  // Toggle mode
        drawModeButton();      // Update button appearance
        return;                // Don't process as canvas touch
      }
    }
    
    // Check if tapped within the 28×28 drawing canvas
    int gridX = (t_x - CANVAS_MARGIN) / BLOCK_SIZE;
    int gridY = (t_y - CANVAS_MARGIN) / BLOCK_SIZE;
    if (!buttonWasPressed && gridX >= 0 && gridX < CANVAS_SIZE &&
        gridY >= 0 && gridY < CANVAS_SIZE) {
      // Draw with the 3x3 square brush using current mode
      draw(gridX, gridY, drawMode);
      drawModeButton();
      
      // Add a small delay to avoid overwhelming the classification task
      delay(10);
      
      // Trigger classification after drawing
      classifyCanvas();
    }
  }
  else {
    if (wasPressed) {
      // If we just finished a stroke, ensure we get a final classification
      if (!buttonWasPressed) {
        while (classificationData.processing) {
          delay(10);
        }  // Wait a bit for any in-progress classification
        classifyCanvas();  // Trigger final classification
      }
    }
    wasPressed = false;
    buttonWasPressed = false;
  }
}

bool needsFinalClassification = false;

void classifyCanvas() {
  // Don't start new classification if one is already processing
  if (classificationData.processing) {
    Serial.println("Skipping classification - already processing");
    needsFinalClassification = true;  // Mark that we need a final classification
    return;
  }
  
  Serial.println("Starting new classification");
  needsFinalClassification = false;  // Clear the flag since we're starting a classification
  
  // Convert the boolean canvas[][] to a float array (28×28)
  for (int y = 0; y < CANVAS_SIZE; y++) {
    for (int x = 0; x < CANVAS_SIZE; x++) {
      // Now true means white, false means black
      classificationData.image[y * CANVAS_SIZE + x] = (canvas[x][y]) ? 1.0f : 0.0f;
    }
  }
  
  // Signal the classification task
  classificationData.pending = true;
}

// Classification task that runs on core 1
void classifyTask(void * parameter) {
  float output[10];
  
  while(true) {
    if (classificationData.pending && !classificationData.processing) {
      Serial.println("Classification task starting inference");
      classificationData.processing = true;
      classificationData.pending = false;
      
      // Run inference
      forwardPass(classificationData.image, output);
      Serial.println("Forward pass complete");
      
      // Find the most likely class
      float bestVal = output[0];
      int bestIdx = 0;
      for (int i = 1; i < 10; i++) {
        if (output[i] > bestVal) {
          bestVal = output[i];
          bestIdx = i;
        }
      }
      
      Serial.printf("Best prediction: %d (%.3f)\n", bestIdx, bestVal);
      
      // Take mutex before accessing TFT
      if (xSemaphoreTake(tftMutex, portMAX_DELAY)) {
        delay(10);

        // Clear a region below the button for text
        tft.fillRect(0, CANVAS_MARGIN + (CANVAS_SIZE * BLOCK_SIZE) + 5, WIDTH, 60, TFT_BLACK);
        tft.setCursor(10, CANVAS_MARGIN + (CANVAS_SIZE * BLOCK_SIZE) + 10);
        tft.setTextSize(2);
        tft.setTextColor(TFT_GREEN, TFT_BLACK);
        tft.printf("Pred: %d (Prob=%.3f)\r\n", bestIdx, bestVal);
        
        // Print each probability
        tft.setCursor(10, CANVAS_MARGIN + (CANVAS_SIZE * BLOCK_SIZE) + 35);
        tft.setTextSize(1);
        tft.setTextColor(TFT_WHITE, TFT_BLACK);
        for (int i = 0; i < 10; i++) {
          tft.printf(" %d:%.3f  ", i, output[i]);
          if (i == 4) {
            tft.setCursor(10, CANVAS_MARGIN + (CANVAS_SIZE * BLOCK_SIZE) + 45);
          }
        }

        delay(10);
        
        // Release the mutex
        xSemaphoreGive(tftMutex);
        Serial.println("Updated display");
      }
      
      classificationData.processing = false;
      
      // If we need a final classification, trigger it now
      if (needsFinalClassification) {
        Serial.println("Triggering missed final classification");
        classifyCanvas();
      }
    }
    // Small delay to prevent tight loop
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

/*******************************************************************************
 *  CANVAS DRAWING FUNCTIONS
 ******************************************************************************/

void drawCanvas() {
  // Take mutex once for the entire canvas drawing operation
  if (xSemaphoreTake(tftMutex, portMAX_DELAY)) {
    for (int y = 0; y < CANVAS_SIZE; y++) {
      for (int x = 0; x < CANVAS_SIZE; x++) {
        tft.fillRect(
          CANVAS_MARGIN + (x * BLOCK_SIZE),
          CANVAS_MARGIN + (y * BLOCK_SIZE),
          BLOCK_SIZE,
          BLOCK_SIZE,
          canvas[x][y] ? TFT_WHITE : TFT_BLACK
        );
      }
    }

    // Draw mode button on top of canvas
    drawModeButton(); 

    xSemaphoreGive(tftMutex);
  }
}

void setPixel(int x, int y, bool color) {
  if (x >= 0 && x < CANVAS_SIZE && y >= 0 && y < CANVAS_SIZE) {
    canvas[x][y] = color;
  }
}

void setBrushPlus(int cx, int cy, bool color) {
  // Center pixel
  setPixel(cx, cy, color);
  
  // Top
  setPixel(cx, cy-1, color);
  // Bottom
  setPixel(cx, cy+1, color);
  // Left
  setPixel(cx-1, cy, color);
  // Right
  setPixel(cx+1, cy, color);
}

void setBrushSquare(int cx, int cy, bool color) {
  for (int y = cy - 1; y <= cy + 1; y++) {
    for (int x = cx - 1; x <= cx + 1; x++) {
      setPixel(x, y, color);
    }
  }
}

void draw(int cx, int cy, bool color) {
  setBrushSquare(cx, cy, color);
  
  // Take mutex once for all drawing operations
  if (xSemaphoreTake(tftMutex, portMAX_DELAY)) {
    // Draw the 3x3 block
    for (int y = cy - 1; y <= cy + 1; y++) {
      if (y >= 0 && y < CANVAS_SIZE) {
        for (int x = cx - 1; x <= cx + 1; x++) {
          if (x >= 0 && x < CANVAS_SIZE) {
            tft.fillRect(
              CANVAS_MARGIN + (x * BLOCK_SIZE),
              CANVAS_MARGIN + (y * BLOCK_SIZE),
              BLOCK_SIZE,
              BLOCK_SIZE,
              color ? TFT_WHITE : TFT_BLACK
            );
          }
        }
      }
    }
    xSemaphoreGive(tftMutex);
  }
}

/*******************************************************************************
 *  TOUCHSCREEN CALIBRATION FUNCTION
 ******************************************************************************/
void touch_calibrate() {
  uint16_t calData[5];
  uint8_t calDataOK = 0;

  Serial.println("Calibrating touch...");

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
    Serial.println("Calibrating touch... 2");
    // Do calibration
    if (xSemaphoreTake(tftMutex, portMAX_DELAY)) {
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
      xSemaphoreGive(tftMutex);
    }
  }
}
