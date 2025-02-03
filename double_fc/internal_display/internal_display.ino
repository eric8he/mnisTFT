#include <TFT_eSPI.h>
#include <WiFi.h>
#include <esp_now.h>

//----------------------
// Display & Canvas Setup
//----------------------
#define WIDTH 320
#define HEIGHT 480
#define CANVAS_SIZE 28
#define BLOCK_SIZE 10
#define CANVAS_MARGIN ((WIDTH - (CANVAS_SIZE * BLOCK_SIZE)) / 2)

//----------------------
// ESP-NOW Packet Setup
//----------------------
#define PAYLOAD_SIZE 196      // Each packet carries 196 bytes (196*4=784 bytes total)
#define NUM_PACKETS 4         // Total packets needed to send a 28x28 (784-byte) canvas

// Global buffer for reassembled canvas (784 bytes)
uint8_t receivedCanvas[CANVAS_SIZE * CANVAS_SIZE];
// Array to track which packet has been received
bool packetReceived[NUM_PACKETS] = { false, false, false, false };
volatile uint8_t packetsCount = 0;
// Flag to signal that the complete canvas has been reassembled
volatile bool updateDisplayFlag = false;

// Define the ESP-NOW packet structure (packed to avoid padding)
typedef struct {
  uint8_t packetId;
  uint8_t totalPackets; // Should be NUM_PACKETS (4)
  uint8_t payload[PAYLOAD_SIZE];
} __attribute__((packed)) espnow_packet_t;

//----------------------
// TFT Display Object
//----------------------
TFT_eSPI tft = TFT_eSPI();

//----------------------
// Function: Draw the Reassembled Canvas
//----------------------
void drawReceivedCanvas() {
  //tft.fillScreen(TFT_BLACK);
  tft.drawRect(CANVAS_MARGIN - 1, CANVAS_MARGIN - 1,
               (CANVAS_SIZE * BLOCK_SIZE) + 2,
               (CANVAS_SIZE * BLOCK_SIZE) + 2, TFT_WHITE);
  
  for (int y = 0; y < CANVAS_SIZE; y++) {
    for (int x = 0; x < CANVAS_SIZE; x++) {
      // Each pixel is represented as 1 (white) or 0 (black)
      uint8_t pixel = receivedCanvas[y * CANVAS_SIZE + x];
      uint16_t color = (pixel == 1) ? TFT_WHITE : TFT_BLACK;
      tft.fillRect(CANVAS_MARGIN + (x * BLOCK_SIZE),
                   CANVAS_MARGIN + (y * BLOCK_SIZE),
                   BLOCK_SIZE, BLOCK_SIZE, color);
    }
  }
  Serial.println("Display updated.");
}

//----------------------
// ESP-NOW Receive Callback with Debug Messages
//----------------------
void onDataRecv(const esp_now_recv_info_t *recv_info, const uint8_t *data, int len) {
  char macStr[18];
  snprintf(macStr, sizeof(macStr), "%02x:%02x:%02x:%02x:%02x:%02x",
           recv_info->src_addr[0], recv_info->src_addr[1],
           recv_info->src_addr[2], recv_info->src_addr[3],
           recv_info->src_addr[4], recv_info->src_addr[5]);
  Serial.print("Packet received from: ");
  Serial.println(macStr);
  
  // Verify the packet length is as expected
  if (len != sizeof(espnow_packet_t)) {
    Serial.print("Unexpected packet length: ");
    Serial.println(len);
    return;
  }
  
  // Copy incoming data into a local packet structure
  espnow_packet_t packet;
  memcpy(&packet, data, sizeof(packet));
  
  // Debug: Print packet header information
  Serial.print("Packet ID: ");
  Serial.println(packet.packetId);
  Serial.print("Total Packets: ");
  Serial.println(packet.totalPackets);
  
  // Debug: Dump the entire payload contents (or just a portion)
  Serial.print("Payload: ");
  for (int i = 0; i < PAYLOAD_SIZE; i++) {
    Serial.print(packet.payload[i], DEC);
    Serial.print(" ");
  }
  Serial.println();
  
  // Copy the received payload into the correct position in receivedCanvas
  if (packet.packetId < NUM_PACKETS) {
    memcpy(receivedCanvas + (packet.packetId * PAYLOAD_SIZE), packet.payload, PAYLOAD_SIZE);
    if (!packetReceived[packet.packetId]) {
      packetReceived[packet.packetId] = true;
      packetsCount++;
    }
  }
  
  // When all packets have been received, print a summary and set flag for display update
  if (packetsCount == NUM_PACKETS) {
    Serial.println("All canvas packets received; reassembled canvas (first 16 bytes):");
    for (int i = 0; i < 16; i++) {
      Serial.print(receivedCanvas[i], DEC);
      Serial.print(" ");
    }
    Serial.println();
    updateDisplayFlag = true;
  }
}

//----------------------
// Setup Function
//----------------------
void setup() {
  Serial.begin(115200);
  tft.init();
  tft.setRotation(0);
  tft.fillScreen(TFT_BLACK);
  
  // Initialize WiFi in Station mode
  WiFi.mode(WIFI_STA);
  
  // Initialize ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }
  
  // Register the receive callback with the updated signature
  esp_now_register_recv_cb(onDataRecv);
  
  Serial.println("ESP-NOW Receiver is running and waiting for canvas data...");
}

//----------------------
// Main Loop
//----------------------
void loop() {
  // Check if the complete canvas has been received
  if (updateDisplayFlag) {
    drawReceivedCanvas();
    // Reset packet tracking for next transmission
    for (int i = 0; i < NUM_PACKETS; i++) {
      packetReceived[i] = false;
    }
    packetsCount = 0;
    updateDisplayFlag = false;
  }
  //delay(10);
}
