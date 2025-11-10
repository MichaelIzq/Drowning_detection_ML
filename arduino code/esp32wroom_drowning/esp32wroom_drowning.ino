#include <Arduino.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>

// ====== PINS (ESP32-S, all on one side) ======
#define SD_SCK   18
#define SD_MISO  19
#define SD_MOSI  23
#define SD_CS     5

#define I2C_SDA  21
#define I2C_SCL  22

// ====== SD CONFIG ======
const uint32_t SD_SPI_HZ = 12000000UL;   // safe starting speed

// ====== LIS3DH CONFIG ======
#define LIS3DH_ADDR 0x18
#define LIS3DH_REG_WHOAMI   0x0F
#define LIS3DH_REG_CTRL1    0x20
#define LIS3DH_REG_CTRL4    0x23
#define LIS3DH_REG_OUT_X_L  0x28

// Sample rate (Hz)
const uint16_t SAMPLE_HZ = 100;

File logFile;

// ---------- LIS3DH helpers ----------
bool lis3dhWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(LIS3DH_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

bool lis3dhReadBytes(uint8_t startReg, uint8_t *buf, uint8_t len) {
  Wire.beginTransmission(LIS3DH_ADDR);
  Wire.write(startReg | 0x80); // auto-increment
  if (Wire.endTransmission(false) != 0) return false;
  Wire.requestFrom((int)LIS3DH_ADDR, (int)len);
  for (uint8_t i = 0; i < len; i++) {
    if (!Wire.available()) return false;
    buf[i] = Wire.read();
  }
  return true;
}

bool lis3dhInit(uint8_t &whoami) {
  if (!lis3dhReadBytes(LIS3DH_REG_WHOAMI, &whoami, 1)) return false;
  if (whoami != 0x33) return false;

  // 100 Hz ODR, XYZ enable
  if (!lis3dhWrite(LIS3DH_REG_CTRL1, 0x57)) return false;
  // ±2 g, High-Resolution
  if (!lis3dhWrite(LIS3DH_REG_CTRL4, 0x08)) return false;
  return true;
}

bool lis3dhReadRaw(int16_t &x, int16_t &y, int16_t &z) {
  uint8_t d[6];
  if (!lis3dhReadBytes(LIS3DH_REG_OUT_X_L, d, 6)) return false;
  int16_t rx = (int16_t)(d[1] << 8 | d[0]);
  int16_t ry = (int16_t)(d[3] << 8 | d[2]);
  int16_t rz = (int16_t)(d[5] << 8 | d[4]);
  x = rx >> 4; y = ry >> 4; z = rz >> 4;
  return true;
}

inline float countsToG(int16_t c) { return c * 0.001f; }

// ---------- Filename helper ----------
String nextAvailableFilename() {
  if (!SD.exists("/LOGS")) SD.mkdir("/LOGS");
  char name[20]; // "/LOGS/LOG000.CSV"
  for (int i = 0; i <= 999; i++) {
    snprintf(name, sizeof(name), "/LOGS/LOG%03d.CSV", i);
    if (!SD.exists(name)) return String(name);
  }
  return String("/LOGS/LOG999.CSV");
}

// ==================== SETUP ====================
void setup() {
  Serial.begin(115200);
  while (!Serial) {}

  Serial.println("\n=== LIS3DH CSV Logger (ESP32-S) ===");

  // I²C (LIS3DH)
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);

  // SD (VSPI)
  SPI.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
  if (!SD.begin(SD_CS, SPI, SD_SPI_HZ)) {
    Serial.println("SD init FAILED. Check wiring/CS/power/format (FAT32).");
    while (true) delay(500);
  }
  Serial.println("SD init OK.");

  // Card info
  uint8_t ct = SD.cardType();
  Serial.print("Card type: ");
  if (ct == CARD_NONE)         Serial.println("None");
  else if (ct == CARD_MMC)     Serial.println("MMC");
  else if (ct == CARD_SD)      Serial.println("SDSC");
  else if (ct == CARD_SDHC)    Serial.println("SDHC/SDXC");
  else                         Serial.println("Unknown");
  Serial.print("Card size (MB): "); Serial.println(SD.cardSize() / (1024ULL * 1024ULL));

  // LIS3DH init
  uint8_t who = 0;
  if (!lis3dhInit(who)) {
    Serial.println("LIS3DH init FAILED. Try address 0x18↔0x19.");
    while (true) delay(500);
  }
  Serial.print("LIS3DH WHOAMI: 0x"); Serial.println(who, HEX);

  // Create log file
  String fname = nextAvailableFilename();
  Serial.print("Creating log file: "); Serial.println(fname);
  logFile = SD.open(fname.c_str(), FILE_WRITE);
  if (!logFile) {
    Serial.println("Failed to create log file. Card locked or read-only?");
    while (true) delay(500);
  }

  // CSV header
  logFile.println("meta,board=ESP32-S,odr_hz=100,range=+/-2g,format=CSV");
  logFile.println("timestamp_ms,ax_g,ay_g,az_g");
  logFile.flush();

  Serial.println("Logging started.");
}

// ==================== LOOP ====================
void loop() {
  static const uint32_t dt = 1000UL / SAMPLE_HZ;
  static uint32_t tPrev = 0;
  static uint16_t linesSinceFlush = 0;

  uint32_t now = millis();
  if (now - tPrev >= dt) {
    tPrev = now;

    int16_t rx, ry, rz;
    if (lis3dhReadRaw(rx, ry, rz)) {
      float ax = countsToG(rx), ay = countsToG(ry), az = countsToG(rz);

      logFile.print(now);
      logFile.print(',');
      logFile.print(ax, 5);
      logFile.print(',');
      logFile.print(ay, 5);
      logFile.print(',');
      logFile.println(az, 5);

      if (++linesSinceFlush >= 50) {
        linesSinceFlush = 0;
        logFile.flush();
      }
    } else {
      logFile.print(now);
      logFile.println(",NaN,NaN,NaN");
    }
  }
  delay(1);
}
