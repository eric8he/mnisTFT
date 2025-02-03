#include "FS.h"        // For SPIFFS
#include <SPI.h>
#include <TFT_eSPI.h>  // Bodmer's TFT library
#include <WiFi.h>
#include <esp_now.h>   // ESP-NOW library

//#include "nn.h"        // forwardPass(...) declaration
//#include "weights.h"   // conv1_weight, fc1_bias, etc.

//--------------------------------------------
// TFT and Touch Setup
//--------------------------------------------
TFT_eSPI tft = TFT_eSPI();

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
#define MODE_BTN_Y (CANVAS_MARGIN + MODE_BTN_MARGIN)

bool drawMode = true;  // true = draw (white), false = erase (black)

void drawModeButton() {
  if (xSemaphoreTake(tftMutex, portMAX_DELAY)) {
    tft.fillRect(MODE_BTN_X, MODE_BTN_Y, MODE_BTN_SIZE, MODE_BTN_SIZE, drawMode ? TFT_WHITE : TFT_BLACK);
    tft.drawRect(MODE_BTN_X, MODE_BTN_Y, MODE_BTN_SIZE, MODE_BTN_SIZE, TFT_WHITE);
    tft.setTextColor(drawMode ? TFT_BLACK : TFT_WHITE);
    tft.setTextSize(2);
    tft.setCursor(MODE_BTN_X + 8, MODE_BTN_Y + 8);
    tft.print(drawMode ? "D" : "E");
    xSemaphoreGive(tftMutex);
  }
}

//--------------------------------------------
// ESP-NOW definitions for sending canvas data
//--------------------------------------------
#define PAYLOAD_SIZE 196   // 196 * 4 = 784 bytes total

typedef struct {
  uint8_t packetId;
  uint8_t totalPackets;
  uint8_t payload[PAYLOAD_SIZE];
} espnow_packet_t;

uint8_t broadcastAddress[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// Function to send the canvas over ESP-NOW
void sendCanvasOverESPNOW() {
  // Create a temporary buffer of 784 bytes.
  uint8_t data[CANVAS_SIZE * CANVAS_SIZE];
  for (int i = 0; i < CANVAS_SIZE * CANVAS_SIZE; i++) {
    // classificationData.image contains 1.0 (white) or 0.0 (black)
    data[i] = (classificationData.image[i] >= 0.5f) ? 1 : 0;
  }
  const uint8_t totalPackets = (CANVAS_SIZE * CANVAS_SIZE) / PAYLOAD_SIZE; // Should be 4
  for (uint8_t packetId = 0; packetId < totalPackets; packetId++) {
    espnow_packet_t packet;
    packet.packetId = packetId;
    packet.totalPackets = totalPackets;
    memcpy(packet.payload, data + (packetId * PAYLOAD_SIZE), PAYLOAD_SIZE);
    esp_err_t result = esp_now_send(broadcastAddress, (uint8_t*)&packet, sizeof(packet));
    if (result == ESP_OK) {
      Serial.printf("Packet %d sent successfully\n", packetId);
    } else {
      Serial.printf("Error sending packet %d: %d\n", packetId, result);
    }
    delay(10); // brief delay between packets
  }
}

//--------------------------------------------
// SETUP
//--------------------------------------------
void setup() {
  Serial.begin(115200);
  while(!Serial) { delay(10); }
  Serial.println("Serial started");
  
  // Create mutex for TFT access before initializing TFT
  tftMutex = xSemaphoreCreateMutex();
  
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
  tft.drawRect(CANVAS_MARGIN - 1, CANVAS_MARGIN - 1,
               (CANVAS_SIZE * BLOCK_SIZE) + 2, (CANVAS_SIZE * BLOCK_SIZE) + 2, TFT_WHITE);

  // Fill canvas area with black
  tft.fillRect(CANVAS_MARGIN, CANVAS_MARGIN,
               CANVAS_SIZE * BLOCK_SIZE, CANVAS_SIZE * BLOCK_SIZE, TFT_BLACK);

  // Initialize the canvas array to all black
  for (int y = 0; y < CANVAS_SIZE; y++) {
    for (int x = 0; x < CANVAS_SIZE; x++) {
      canvas[x][y] = false;
    }
  }
  
  drawModeButton();
  
  classificationData.pending = false;
  classificationData.processing = false;
  
  // Create classification task on core 1
  // xTaskCreatePinnedToCore(
  //   classifyTask,        // classification task function (already in your code)
  //   "classifyTask",      
  //   8192,               
  //   NULL,               
  //   1,                  
  //   &classifyTaskHandle,
  //   1                   
  // );
  
  // --- Initialize ESP-NOW ---
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
  }
  // Add broadcast peer
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add broadcast peer");
  }
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
    if (!wasPressed) {
      wasPressed = true;
      // Check for mode button press
      if (t_x >= MODE_BTN_X && t_x < (MODE_BTN_X + MODE_BTN_SIZE) &&
          t_y >= MODE_BTN_Y && t_y < (MODE_BTN_Y + MODE_BTN_SIZE)) {
        buttonWasPressed = true;
        drawMode = !drawMode;
        drawModeButton();
        return;
      }
    }
    int gridX = (t_x - CANVAS_MARGIN) / BLOCK_SIZE;
    int gridY = (t_y - CANVAS_MARGIN) / BLOCK_SIZE;
    if (!buttonWasPressed && gridX >= 0 && gridX < CANVAS_SIZE &&
        gridY >= 0 && gridY < CANVAS_SIZE) {
      draw(gridX, gridY, drawMode);
      drawModeButton();
      delay(10);
      classifyCanvas();
    }
  }
  else {
    if (wasPressed && !buttonWasPressed) {
      while (classificationData.processing) { delay(10); }
      classifyCanvas();
    }
    wasPressed = false;
    buttonWasPressed = false;
  }
}

bool needsFinalClassification = false;

void classifyCanvas() {
  // if (classificationData.processing) {
  //   Serial.println("Skipping classification - already processing");
  //   needsFinalClassification = true;
  //   return;
  // }
  
  // Serial.println("Starting new classification");
  // needsFinalClassification = false;
  
  for (int y = 0; y < CANVAS_SIZE; y++) {
    for (int x = 0; x < CANVAS_SIZE; x++) {
      classificationData.image[y * CANVAS_SIZE + x] = canvas[x][y] ? 1.0f : 0.0f;
    }
  }
  
  // // Signal the classification task
  // classificationData.pending = true;
  
  // --- Send the input array over ESP-NOW ---
  sendCanvasOverESPNOW();
}



// Classification task that runs on core 1
void classifyTask(void * parameter) {
  float output[10];
  
  // while(true) {
  //   if (classificationData.pending && !classificationData.processing) {
  //     Serial.println("Classification task starting inference");
  //     classificationData.processing = true;
  //     classificationData.pending = false;
      
  //     // Run inference
  //     forwardPass(classificationData.image, output);
  //     Serial.println("Forward pass complete");
      
  //     // Find the most likely class
  //     float bestVal = output[0];
  //     int bestIdx = 0;
  //     for (int i = 1; i < 10; i++) {
  //       if (output[i] > bestVal) {
  //         bestVal = output[i];
  //         bestIdx = i;
  //       }
  //     }
      
  //     Serial.printf("Best prediction: %d (%.3f)\n", bestIdx, bestVal);
      
  //     // Take mutex before accessing TFT
  //     if (xSemaphoreTake(tftMutex, portMAX_DELAY)) {
  //       delay(10);

  //       // Clear a region below the button for text
  //       tft.fillRect(0, CANVAS_MARGIN + (CANVAS_SIZE * BLOCK_SIZE) + 5, WIDTH, 60, TFT_BLACK);
  //       tft.setCursor(10, CANVAS_MARGIN + (CANVAS_SIZE * BLOCK_SIZE) + 10);
  //       tft.setTextSize(2);
  //       tft.setTextColor(TFT_GREEN, TFT_BLACK);
  //       tft.printf("Pred: %d (Prob=%.3f)\r\n", bestIdx, bestVal);
        
  //       // Print each probability
  //       tft.setCursor(10, CANVAS_MARGIN + (CANVAS_SIZE * BLOCK_SIZE) + 35);
  //       tft.setTextSize(1);
  //       tft.setTextColor(TFT_WHITE, TFT_BLACK);
  //       for (int i = 0; i < 10; i++) {
  //         tft.printf(" %d:%.3f  ", i, output[i]);
  //         if (i == 4) {
  //           tft.setCursor(10, CANVAS_MARGIN + (CANVAS_SIZE * BLOCK_SIZE) + 45);
  //         }
  //       }

  //       delay(10);
        
  //       // Release the mutex
  //       xSemaphoreGive(tftMutex);
  //       Serial.println("Updated display");
  //     }
      
  //     classificationData.processing = false;
      
  //     // If we need a final classification, trigger it now
  //     if (needsFinalClassification) {
  //       Serial.println("Triggering missed final classification");
  //       classifyCanvas();
  //     }
  //   }
  //   // Small delay to prevent tight loop
  //   vTaskDelay(pdMS_TO_TICKS(10));
  // }
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

